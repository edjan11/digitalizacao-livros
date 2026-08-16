"""Auditoria e importacao retomavel de livros organizados por face."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Callable

import cv2
import numpy as np

from ..database.repository import Repository
from ..duplicate.hashing import compute_hashes
from ..imaging.document import retificar_formulario
from ..imaging.page_orientation import OrientationDetector, rotate_image
from ..imaging.book_layouts import layout_for_book


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PageResolver = Callable[[Path, str], int | None]
A16_MISSING_FACES = frozenset({(27, "frente"), (37, "frente"), (5, "verso")})


@dataclass(frozen=True)
class BookImportSpec:
    codigo: str
    acervo_id: int
    oficio_id: int
    tipo_id: int
    total_folhas: int
    termo_inicial: int
    termo_final: int
    registros_por_face: int = 2
    nome_capa: str = ""

    @classmethod
    def a16(cls) -> "BookImportSpec":
        return cls(
            codigo="A-16",
            acervo_id=12,
            oficio_id=12,
            tipo_id=1,
            total_folhas=300,
            termo_inicial=16803,
            termo_final=18002,
            registros_por_face=2,
            nome_capa="Livro A nº 16 — Nascimentos — 12º Ofício",
        )

    def terms(self, folha: int, face: str) -> tuple[int, int]:
        offset = (int(folha) - 1) * 4 + (0 if face == "frente" else 2)
        start = self.termo_inicial + offset
        return start, start + 1


@dataclass(frozen=True)
class AuditItem:
    path: Path | None
    sha256: str
    pasta_origem: str
    tipo_documento: str
    folha: int | None
    face: str
    termo_inicial: int | None
    termo_final: int | None
    status: str = "auditado"
    erro: str = ""


@dataclass(frozen=True)
class BookAudit:
    root: Path
    spec: BookImportSpec
    registros: tuple[AuditItem, ...]
    indices: tuple[Path, ...]
    faltantes: tuple[AuditItem, ...]
    duplicados: tuple[tuple[Path, Path], ...]
    nao_resolvidos: tuple[Path, ...]

    @property
    def total_files(self) -> int:
        return len(self.registros) + len(self.indices) + len(self.nao_resolvidos)

    @property
    def ready(self) -> bool:
        return not self.nao_resolvidos

    def to_dict(self) -> dict:
        def item(value: AuditItem) -> dict:
            result = asdict(value)
            result["path"] = str(value.path) if value.path else None
            return result
        return {
            "root": str(self.root),
            "spec": asdict(self.spec),
            "registros": [item(value) for value in self.registros],
            "indices": [str(value) for value in self.indices],
            "faltantes": [item(value) for value in self.faltantes],
            "duplicados": [[str(a), str(b)] for a, b in self.duplicados],
            "nao_resolvidos": [str(value) for value in self.nao_resolvidos],
            "ready": self.ready,
        }


def _normalized(value: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    ).upper()


def _folder(root: Path, aliases: set[str]) -> Path | None:
    return next(
        (child for child in root.iterdir() if child.is_dir() and _normalized(child.name) in aliases),
        None,
    )


def _images(folder: Path | None) -> list[Path]:
    if folder is None:
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class RapidPageResolver:
    """Le numero impresso da folha ou um termo compativel com a face."""

    def __init__(self, spec: BookImportSpec) -> None:
        self.spec = spec
        self._engine = None

    def _ocr(self, image: np.ndarray) -> str:
        if self._engine is None:
            from ..ocr.rapidocr_engine import RapidOCREngine
            self._engine = RapidOCREngine()
        h, w = image.shape[:2]
        # O primeiro cabeçalho contém a folha impressa e o primeiro termo. Um
        # recorte curto evita datas/horas do restante do assento e é muito mais
        # rápido que OCR da meia página.
        crop = image[: max(1, int(h * 0.27)), :]
        scale = min(1.0, 1200 / max(crop.shape[1], 1))
        if scale < 1:
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return self._engine.read_array(crop)

    @staticmethod
    def _distance(a: str, b: str) -> int:
        previous = list(range(len(b) + 1))
        for index, ca in enumerate(a, 1):
            current = [index]
            for column, cb in enumerate(b, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ca != cb),
                ))
            previous = current
        return previous[-1]

    def resolve_expected(
        self, path: Path, face: str, expected_leaves: range | list[int] | tuple[int, ...]
    ) -> int | None:
        image = _read(path)
        if image is None:
            return None
        text = self._ocr(image)
        normalized = _normalized(text)
        form_markers = sum(
            marker in normalized
            for marker in ("NUMERO", "MIL NOVECENTOS", "CARTORIO", "RECEBEU O NOME", "NASCEU UMA CRIANCA")
        )
        if form_markers < 1:
            return None
        raw_tokens = re.findall(r"(?<!\d)[\d.\-/]{3,7}(?!\d)", text)
        observed = [re.sub(r"\D", "", token) for token in raw_tokens]
        observed = [token for token in observed if len(token) >= 3]
        best: tuple[float, int] | None = None
        for leaf in expected_leaves:
            if not 1 <= int(leaf) <= self.spec.total_folhas:
                continue
            terms = self.spec.terms(int(leaf), face)
            targets = [f"{int(leaf):04d}"]
            for term in terms:
                value = str(term)
                targets.extend((value, value[-4:], value[-3:]))
            for token in observed:
                for target in targets:
                    # Para termos truncados, o sufixo exato é evidência forte;
                    # para o número impresso aceitamos no máximo uma troca.
                    distance = self._distance(token[-len(target):], target)
                    penalty = distance / max(len(target), 1)
                    if token.endswith(target):
                        penalty = -0.05
                    candidate = (penalty, int(leaf))
                    if best is None or candidate < best:
                        best = candidate
        return best[1] if best is not None and best[0] <= 0.20 else None

    def __call__(self, path: Path, face: str) -> int | None:
        image = _read(path)
        if image is None:
            return None
        text = self._ocr(image)
        numbers = [int(value) for value in re.findall(r"(?<!\d)\d{1,5}(?!\d)", text)]
        # A foliacao impressa (0001..0300) e a evidencia preferida.
        leaves = [value for value in numbers if 1 <= value <= self.spec.total_folhas]
        if leaves:
            return leaves[0]
        for value in numbers:
            if not self.spec.termo_inicial <= value <= self.spec.termo_final:
                continue
            delta = value - self.spec.termo_inicial
            expected_mod = {0, 1} if face == "frente" else {2, 3}
            if delta % 4 in expected_mod:
                return delta // 4 + 1
        return None


def _map_ordered_sequence(
    paths: list[Path], face: str, resolver: RapidPageResolver, total_leaves: int,
    *, trusted_endpoints: bool = False,
) -> tuple[dict[Path, int], list[Path]]:
    """Localiza saltos sem executar OCR em todas as faces.

    Com N arquivos para T folhas, cada posição só pode representar a própria
    posição ou uma das T-N seguintes. Âncoras e busca binária acham onde esse
    deslocamento aumenta; nenhuma ausência desloca o restante por palpite.
    """
    missing_count = max(0, total_leaves - len(paths))
    cache: dict[int, int | None] = {}

    def resolve(position: int) -> int | None:
        if position not in cache:
            start = position + 1
            cache[position] = resolver.resolve_expected(
                paths[position], face,
                range(start, min(total_leaves, start + missing_count) + 1),
            )
        return cache[position]

    # Provas obrigatórias das extremidades.
    first, last = resolve(0), resolve(len(paths) - 1)
    if trusted_endpoints:
        # Usado somente após conferência visual explícita das quatro pontas e
        # registrado no relatório da auditoria.
        first, last = 1, total_leaves
        cache[0], cache[len(paths) - 1] = first, last
    if first != 1 or last != total_leaves:
        return {}, list(paths)
    boundaries: list[int] = []
    low_position = 0
    low_offset = 0
    for target_offset in range(1, missing_count + 1):
        lo, hi = low_position + 1, len(paths) - 1
        found = None
        while lo <= hi:
            mid = (lo + hi) // 2
            leaf = resolve(mid)
            if leaf is None:
                # Busca local pequena; se não houver evidência, a auditoria
                # permanece inconclusiva em vez de inferir silenciosamente.
                alternatives = [p for delta in range(1, 5) for p in (mid - delta, mid + delta)
                                if lo <= p <= hi]
                evidence = next(((p, resolve(p)) for p in alternatives if resolve(p) is not None), None)
                if evidence is None:
                    break
                mid, leaf = evidence
            offset = int(leaf) - (mid + 1)
            if offset >= target_offset:
                found = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if found is None:
            return {}, list(paths)
        # Refina linearmente ao redor do limite para exigir uma âncora dos dois lados.
        start = max(low_position + 1, found - 8)
        end = min(len(paths) - 1, found + 8)
        confirmed = None
        for position in range(start, end + 1):
            leaf = resolve(position)
            if leaf is not None and leaf - (position + 1) >= target_offset:
                confirmed = position
                break
        if confirmed is None:
            return {}, list(paths)
        boundaries.append(confirmed)
        low_position, low_offset = confirmed, target_offset

    mapping: dict[Path, int] = {}
    offset = 0
    boundary_index = 0
    for position, path in enumerate(paths):
        while boundary_index < len(boundaries) and position >= boundaries[boundary_index]:
            offset += 1
            boundary_index += 1
        mapping[path] = position + 1 + offset
    return mapping, []


def auditar_livro(
    root: str | Path,
    spec: BookImportSpec,
    *,
    page_resolver: PageResolver | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> BookAudit:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Pasta do livro nao encontrada: {root}")
    front_dir = _folder(root, {"FRENTE", "FRONT"})
    back_dir = _folder(root, {"VERSO", "BACK"})
    index_dir = _folder(root, {"INDICE", "INDECE", "INDEX"})
    if front_dir is None or back_dir is None:
        raise ValueError("O livro precisa conter as pastas frente e VERSO.")
    front = _images(front_dir)
    back = _images(back_dir)
    indices = list(_images(index_dir))
    resolver = page_resolver or RapidPageResolver(spec)
    built_in = page_resolver is None
    if built_in and spec.codigo == "A-16":
        # Exceção comprovada na auditoria física: esta página alfabética foi
        # fotografada ao final de `frente`; classificá-la é lógico e não move o JPG.
        misplaced = next(
            (path for path in front if path.name == "IMG_2025_08_07_14_32_15S.jpg"), None
        )
        if misplaced:
            front.remove(misplaced)
            indices.append(misplaced)
    sequence_maps: dict[str, dict[Path, int]] = {}
    sequence_unresolved: list[Path] = []
    if built_in:
        for paths, face in ((front, "frente"), (back, "verso")):
            if spec.codigo == "A-16":
                expected = [
                    leaf for leaf in range(1, spec.total_folhas + 1)
                    if (leaf, face) not in A16_MISSING_FACES
                ]
                if len(expected) != len(paths):
                    mapping, failures = {}, list(paths)
                else:
                    mapping = dict(zip(paths, expected))
                    failures = []
            else:
                mapping, failures = _map_ordered_sequence(
                    paths, face, resolver, spec.total_folhas,
                )
            sequence_maps[face] = mapping
            sequence_unresolved.extend(failures)
    source = [(path, "frente") for path in front] + [(path, "verso") for path in back]
    total = len(source)
    mapped: dict[tuple[int, str], AuditItem] = {}
    unresolved: list[Path] = []
    duplicates: list[tuple[Path, Path]] = []
    hashes: dict[str, Path] = {}
    for position, (path, face) in enumerate(source, 1):
        digest = _sha(path)
        if digest in hashes:
            duplicates.append((hashes[digest], path))
        else:
            hashes[digest] = path
        folha = sequence_maps.get(face, {}).get(path) if built_in else resolver(path, face)
        if folha is None or not 1 <= int(folha) <= spec.total_folhas:
            # Um arquivo sem folha, colocado em FRENTE, e tratado logicamente
            # como indice. Ele permanece byte a byte na pasta original.
            if face == "frente" and path not in sequence_unresolved:
                indices.append(path)
            else:
                unresolved.append(path)
            if on_progress:
                on_progress(position, total, f"Sem folha: {path.name}")
            continue
        key = (int(folha), face)
        terms = spec.terms(int(folha), face)
        item = AuditItem(
            path=path,
            sha256=digest,
            pasta_origem=path.parent.name,
            tipo_documento="registro",
            folha=int(folha),
            face=face,
            termo_inicial=terms[0],
            termo_final=terms[1],
        )
        if key in mapped:
            duplicates.append((mapped[key].path, path))
            unresolved.append(path)
        else:
            mapped[key] = item
        if on_progress:
            on_progress(position, total, f"Folha {folha} {face}")

    missing: list[AuditItem] = []
    for folha in range(1, spec.total_folhas + 1):
        for face in ("frente", "verso"):
            if (folha, face) not in mapped:
                terms = spec.terms(folha, face)
                missing.append(AuditItem(
                    path=None,
                    sha256="",
                    pasta_origem=face,
                    tipo_documento="ausente",
                    folha=folha,
                    face=face,
                    termo_inicial=terms[0],
                    termo_final=terms[1],
                    status="faltante",
                    erro="face ausente; sequencia preservada",
                ))
    return BookAudit(
        root=root,
        spec=spec,
        registros=tuple(sorted(mapped.values(), key=lambda item: (item.folha or 0, item.face))),
        indices=tuple(sorted(set(indices))),
        faltantes=tuple(missing),
        duplicados=tuple(duplicates),
        nao_resolvidos=tuple(unresolved),
    )


class GenericBookImporter:
    def __init__(
        self,
        repo: Repository,
        *,
        normalized_root: Path,
        orientation_detector: OrientationDetector | None = None,
    ) -> None:
        self.repo = repo
        self.normalized_root = Path(normalized_root)
        self.orientation_detector = orientation_detector or OrientationDetector()

    def _book(self, spec: BookImportSpec) -> int:
        current = self.repo.db.fetchone(
            "SELECT * FROM livro WHERE acervo_id=? AND codigo=? ORDER BY id LIMIT 1",
            (spec.acervo_id, spec.codigo),
        )
        fields = dict(
            acervo_id=spec.acervo_id,
            oficio_id=spec.oficio_id,
            tipo_id=spec.tipo_id,
            codigo=spec.codigo,
            nome_capa=spec.nome_capa or f"Livro {spec.codigo}",
            total_folhas=spec.total_folhas,
            primeira_folha=1,
            ultima_folha=spec.total_folhas,
            frente_verso=1,
            registros_por_face=spec.registros_por_face,
            termo_inicial=spec.termo_inicial,
            termo_final=spec.termo_final,
            layout_id=(layout_for_book(spec.codigo).layout_id
                       if layout_for_book(spec.codigo) else f"{spec.codigo.lower()}-pendente-v1"),
            status="importando",
            observacoes="Importacao persistente por hash e folha auditada.",
        )
        if current:
            self.repo.atualizar_livro(int(current["id"]), **fields)
            return int(current["id"])
        return self.repo.criar_livro(**fields)

    def _lot(self, audit: BookAudit, livro_id: int) -> int:
        key = f"import-{audit.spec.codigo.lower()}-{hashlib.sha256(str(audit.root).lower().encode()).hexdigest()[:16]}"
        now = datetime.now().isoformat()
        current = self.repo.db.fetchone("SELECT id FROM importacao_lote WHERE chave=?", (key,))
        manifest = json.dumps(audit.to_dict(), ensure_ascii=False)
        if current:
            lot_id = int(current["id"])
            self.repo.db.update(
                "UPDATE importacao_lote SET livro_id=?, total_arquivos=?, manifest_json=?, updated_at=? WHERE id=?",
                (livro_id, audit.total_files, manifest, now, lot_id),
            )
            return lot_id
        return self.repo.db.insert(
            """INSERT INTO importacao_lote
               (chave, livro_id, codigo_livro, raiz, status, total_arquivos,
                manifest_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'auditado', ?, ?, ?, ?)""",
            (key, livro_id, audit.spec.codigo, str(audit.root), audit.total_files, manifest, now, now),
        )

    def _normalized(self, item: AuditItem, book_code: str) -> tuple[dict, np.ndarray]:
        image = _read(item.path)
        if image is None:
            raise ValueError(f"Imagem invalida: {item.path}")
        orientation = self.orientation_detector.detect(image)
        values = {
            # Sugestão incerta nunca altera a forma como os consumidores abrem
            # a foto. A rotação proposta fica apenas no JSON de revisão.
            "rotacao_visualizacao": orientation.rotation if orientation.auto_apply else 0,
            "orientacao_confianca": orientation.confidence,
            "orientacao_metodo": orientation.method,
            "orientacao_motivo": orientation.reason,
            "precisa_revisao": 0 if orientation.auto_apply else 1,
        }
        if not orientation.auto_apply:
            values["normalizacao_json"] = json.dumps({
                "status": "pausado_orientacao",
                "rotacao_sugerida": orientation.rotation,
                "scores": orientation.scores,
            }, ensure_ascii=False)
            return values, image
        upright = rotate_image(image, orientation.rotation)
        rectified = retificar_formulario(upright)
        normalized = rectified.image
        target_dir = self.normalized_root / book_code
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{item.sha256}-layout-v1.jpg"
        if not target.exists():
            ok, encoded = cv2.imencode(".jpg", normalized, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise ValueError("Falha ao criar derivada normalizada")
            encoded.tofile(str(target))
        normalized_sha = _sha(target)
        values.update({
            "caminho_normalizado": str(target),
            "sha256_normalizado": normalized_sha,
            "caminho_armazenamento": str(target),
            "sha256_armazenamento": normalized_sha,
            "normalizacao_json": json.dumps({
                "version": "layout-v1",
                "rotation": orientation.rotation,
                "rectification_applied": rectified.applied,
                "rectification_confidence": rectified.confidence,
                "angle_degrees": rectified.angle_degrees,
                "reason": rectified.reason,
            }, ensure_ascii=False),
        })
        return values, normalized

    def importar(
        self,
        audit: BookAudit,
        *,
        create_derivatives: bool = True,
        on_progress: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        if audit.nao_resolvidos:
            raise ValueError(
                f"Auditoria possui {len(audit.nao_resolvidos)} face(s) sem folha; revise antes de importar."
            )
        livro_id = self._book(audit.spec)
        lot_id = self._lot(audit, livro_id)
        new_images = 0
        interrupted = False
        total = len(audit.registros) + len(audit.indices)
        position = 0
        for item in audit.registros:
            if should_stop and should_stop():
                interrupted = True
                break
            position += 1
            item_status = "pendente_orientacao" if not create_derivatives else "orientado"
            existing = self.repo.db.fetchone(
                "SELECT id FROM imagem WHERE livro_id=? AND sha256=? ORDER BY id LIMIT 1",
                (livro_id, item.sha256),
            )
            if existing:
                image_id = int(existing["id"])
                if create_derivatives:
                    current_image = self.repo.db.fetchone(
                        "SELECT * FROM imagem WHERE id=?", (image_id,)
                    ) or {}
                    if not current_image.get("caminho_normalizado"):
                        normalized_values, _display = self._normalized(item, audit.spec.codigo)
                        if normalized_values:
                            sets = ", ".join(f"{key}=?" for key in normalized_values)
                            self.repo.db.update(
                                f"UPDATE imagem SET {sets} WHERE id=?",
                                tuple(normalized_values.values()) + (image_id,),
                            )
                        if normalized_values.get("precisa_revisao"):
                            item_status = "revisar_orientacao"
                            if not self.repo.tem_revisao_pendente(image_id, "orientacao_incerta"):
                                self.repo.criar_revisao(
                                    imagem_id=image_id,
                                    tipo="orientacao_incerta",
                                    detalhes=(normalized_values.get("orientacao_motivo")
                                              or "Orientacao abaixo de 85%"),
                                )
                    elif float(current_image.get("orientacao_confianca") or 0) < 0.85:
                        item_status = "revisar_orientacao"
            else:
                image = _read(item.path)
                if image is None:
                    raise ValueError(f"Imagem invalida: {item.path}")
                phash, dhash = compute_hashes(image)
                normalized_values = {
                    "rotacao_visualizacao": 0,
                    "orientacao_confianca": 0.0,
                    "orientacao_metodo": "pendente_background",
                    "orientacao_motivo": "Aguardando avaliacao 0/90/180/270",
                    "precisa_revisao": 1,
                    "normalizacao_json": json.dumps({"status": "pendente_orientacao"}),
                }
                display = image
                if create_derivatives:
                    normalized_values, display = self._normalized(item, audit.spec.codigo)
                layout = layout_for_book(audit.spec.codigo)
                image_id = self.repo.registrar_imagem(
                    livro_id=livro_id,
                    ordem_captura=int(item.folha) * 2 - (1 if item.face == "frente" else 0),
                    caminho_original=str(item.path),
                    caminho_thumb=normalized_values.get("caminho_normalizado") or str(item.path),
                    sha256=item.sha256,
                    hash_perceptual=phash,
                    dhash=dhash,
                    tipo_documento="registro",
                    origem_posicao=item.folha,
                    folha_estimada=item.folha,
                    face=item.face,
                    folha_status="confirmado_auditoria",
                    termo_inicial=item.termo_inicial,
                    termo_final=item.termo_final,
                    termo_final_decidido=item.termo_final,
                    folha_final_decidida=item.folha,
                    termo_status="confirmado_auditoria",
                    motor_utilizado="sequencia_auditada_a16",
                    confianca_termo=1.0,
                    confianca_folha=1.0,
                    duplicidade_status="unico",
                    registros_detectados=audit.spec.registros_por_face,
                    layout_id=(layout.layout_id if layout else f"{audit.spec.codigo.lower()}-pendente-v1"),
                    layout_confidence=(layout.confidence if layout else 0.0),
                    layout_method=(layout.calibration if layout else "aguardando_calibracao"),
                    layout_reason=("gabarito exclusivo confirmado em cinco faces"
                                   if layout else "OCR bloqueado ate validar cinco faces"),
                    qualidade_status="pendente",
                    status="aceita" if not normalized_values.get("precisa_revisao") else "revisar_orientacao",
                    **normalized_values,
                )
                new_images += 1
                for registro in self.repo.sincronizar_registros_imagem(image_id):
                    self.repo.salvar_metadado_tratado(
                        imagem_id=image_id,
                        registro_id=int(registro["id"]),
                        tipo="termo",
                        valor=str(registro["termo"]),
                        confianca=1.0,
                        fonte="sequencia_auditada",
                        motor="auditoria_livro_a16",
                        status="confirmado",
                        contexto=f"{audit.spec.codigo}: folha {item.folha} {item.face}",
                    )
                if normalized_values.get("precisa_revisao"):
                    item_status = "revisar_orientacao"
                    self.repo.criar_revisao(
                        imagem_id=image_id,
                        tipo="orientacao_incerta",
                        detalhes=normalized_values.get("orientacao_motivo") or "Orientacao abaixo de 85%",
                    )
            self._upsert_item(lot_id, item, image_id, item_status)
            if on_progress:
                on_progress(position, total, f"Folha {item.folha} {item.face}")

        for index, path in enumerate(audit.indices, 1):
            if interrupted or (should_stop and should_stop()):
                interrupted = True
                break
            position += 1
            digest = _sha(path)
            existing = self.repo.db.fetchone(
                "SELECT id FROM imagem WHERE livro_id=? AND sha256=? ORDER BY id LIMIT 1",
                (livro_id, digest),
            )
            if existing:
                image_id = int(existing["id"])
            else:
                image = _read(path)
                if image is None:
                    continue
                phash, dhash = compute_hashes(image)
                image_id = self.repo.registrar_imagem(
                    livro_id=livro_id, ordem_captura=20_000 + index,
                    caminho_original=str(path), caminho_thumb=str(path), sha256=digest,
                    hash_perceptual=phash, dhash=dhash, tipo_documento="indice",
                    origem_posicao=index, face="indeterminado", qualidade_status="nao_aplicavel",
                    duplicidade_status="unico", status="indice",
                )
                new_images += 1
            self._upsert_item(lot_id, AuditItem(
                path=path, sha256=digest, pasta_origem=path.parent.name,
                tipo_documento="indice", folha=None, face="indeterminado",
                termo_inicial=None, termo_final=None,
            ), image_id, "importado")
            if on_progress:
                on_progress(position, total, f"Indice {index}")

        for missing in audit.faltantes:
            occurrence = self.repo.db.fetchone(
                """SELECT id FROM ocorrencia
                   WHERE livro_id=? AND tipo='face_ausente' AND folha_afetada=?
                     AND termo_afetado=? LIMIT 1""",
                (livro_id, missing.folha, missing.termo_inicial),
            )
            if not occurrence:
                self.repo.criar_ocorrencia(
                    livro_id=livro_id, tipo="face_ausente", folha_afetada=missing.folha,
                    termo_afetado=missing.termo_inicial,
                    descricao=(f"Face {missing.face} ausente; termos {missing.termo_inicial}–"
                               f"{missing.termo_final}. A sequencia seguinte nao foi deslocada."),
                    confirmada=1,
                )
        records = self.repo.db.fetchone(
            "SELECT COUNT(*) n FROM registro WHERE livro_id=?", (livro_id,)
        )["n"]
        status = (
            "importando" if interrupted
            else "precisa_complementacao" if audit.faltantes else "importado"
        )
        self.repo.atualizar_livro(livro_id, status=status, registros_detectados=records)
        orientation_pending = int((self.repo.db.fetchone(
            """SELECT COUNT(*) n FROM importacao_item
               WHERE lote_id=? AND tipo_documento='registro'
                 AND status IN ('pendente_orientacao','revisar_orientacao')""",
            (lot_id,),
        ) or {"n": 0})["n"])
        lot_status = (
            "pausado" if interrupted
            else "aguardando_orientacao" if not create_derivatives
            else "revisar_orientacao" if orientation_pending else "concluido"
        )
        self.repo.db.update(
            "UPDATE importacao_lote SET status=?, processados=?, updated_at=? WHERE id=?",
            (lot_status, position, datetime.now().isoformat(), lot_id),
        )
        self.repo.criar_ou_sincronizar_lote_nomes(livro_id)
        return {
            "livro_id": livro_id,
            "lote_id": lot_id,
            "novas_imagens": new_images,
            "total_imagens": total,
            "registros": int(records),
            "orientacoes_pendentes": orientation_pending,
            "interrompido": interrupted,
            "faltantes": [asdict(item) for item in audit.faltantes],
        }

    def _upsert_item(self, lot_id: int, item: AuditItem, image_id: int, status: str) -> None:
        now = datetime.now().isoformat()
        self.repo.db.update(
            """INSERT INTO importacao_item
               (lote_id, caminho_original, sha256, pasta_origem, tipo_documento,
                folha, face, termo_inicial, termo_final, status, imagem_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(lote_id, caminho_original) DO UPDATE SET
                 sha256=excluded.sha256, pasta_origem=excluded.pasta_origem,
                 tipo_documento=excluded.tipo_documento, folha=excluded.folha,
                 face=excluded.face, termo_inicial=excluded.termo_inicial,
                 termo_final=excluded.termo_final, status=excluded.status,
                 imagem_id=excluded.imagem_id, erro=NULL, updated_at=excluded.updated_at""",
            (
                lot_id, str(item.path), item.sha256, item.pasta_origem,
                item.tipo_documento, item.folha, item.face, item.termo_inicial,
                item.termo_final, status, image_id, now, now,
            ),
        )
