"""Importacao conservadora de livros fotografados em pastas organizadas.

O importador referencia os JPGs originais: nao move, apaga nem recodifica as
fotos. A auditoria do A-07 foi feita sobre a sequencia completa antes de o
livro ser aceito no banco.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata
from typing import Callable

import cv2
import numpy as np

from ..database.repository import Repository
from ..duplicate.hashing import compute_hashes, compute_sha256
from ..imaging.quality import avaliar_qualidade


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Posicoes cronologicas (base 1) confirmadas por pontos visuais/SIFT e por
# leitura humana das duas celulas Numero. O item em A07_KEEP e a melhor foto
# preservada; as demais continuam no banco como recapturas rejeitadas.
A07_DUPLICATE_GROUPS = (
    (151, 152),
    (178, 179),
    (180, 181),
    (188, 189, 190),
    (219, 220),
    (221, 222, 223),
    (248, 249),
    (292, 293),
)
A07_KEEP = {151, 179, 181, 189, 220, 222, 248, 293}
A07_MISSING = (
    (20, 6879, 6880),
    (108, 7231, 7232),
    (243, 7771, 7772),
)


@dataclass(frozen=True)
class A07Audit:
    raiz: Path
    capa: Path
    frentes: tuple[Path, ...]
    versos: tuple[Path, ...]
    indices: tuple[Path, ...]
    termos_por_posicao_verso: dict[int, tuple[int, int]]
    posicoes_rejeitadas: frozenset[int]

    @property
    def total_faces_uteis(self) -> int:
        return len(self.frentes) + len(self.versos) - len(self.posicoes_rejeitadas)


def _images(folder: Path, recursive: bool = False) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _sem_acentos(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    ).upper()


def _read_image(path: Path):
    """Le JPGs com caminhos Unicode no Windows (ex.: pasta INDÍCE)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def auditar_a07(root: str | Path) -> A07Audit:
    root = Path(root)
    frente_dir = root / "FRENTE"
    verso_dir = root / "VERSO"
    if not frente_dir.is_dir() or not verso_dir.is_dir():
        raise ValueError("A pasta precisa conter FRENTE e VERSO.")
    frente_files = _images(frente_dir)
    verso_files = _images(verso_dir)
    if len(frente_files) != 301:
        raise ValueError(
            f"A-07 esperado: 301 fotos em FRENTE (1 abertura + 300 faces); encontrado: {len(frente_files)}."
        )
    if len(verso_files) != 308:
        raise ValueError(
            f"A-07 esperado: 308 fotos em VERSO (307 tentativas + 1 índice); encontrado: {len(verso_files)}."
        )

    index_dir = next(
        (child for child in root.iterdir() if child.is_dir() and _sem_acentos(child.name) == "INDICE"),
        None,
    )
    extra_indices = _images(index_dir, recursive=True) if index_dir else []
    indices = [verso_files[-1], *extra_indices]
    if len(indices) != 29:
        raise ValueError(f"A-07 esperado: 29 fotos de índice; encontrado: {len(indices)}.")

    group_by_position = {
        position: group
        for group in A07_DUPLICATE_GROUPS
        for position in group
    }
    rejected = frozenset(
        position
        for group in A07_DUPLICATE_GROUPS
        for position in group
        if position not in A07_KEEP
    )
    missing_starts = {start for _, start, _ in A07_MISSING}
    terms: dict[int, tuple[int, int]] = {}
    current = 6803
    position = 1
    while position <= 307:
        group = group_by_position.get(position)
        if group and position != group[0]:
            position += 1
            continue
        if current in missing_starts:
            current += 4
        if group:
            for member in group:
                terms[member] = (current, current + 1)
            position = group[-1] + 1
        else:
            terms[position] = (current, current + 1)
            position += 1
        current += 4
    if current != 8003:
        raise AssertionError(f"Sequencia auditada do A-07 terminou em {current}, esperado 8003.")

    return A07Audit(
        raiz=root,
        capa=frente_files[0],
        frentes=tuple(frente_files[1:]),
        versos=tuple(verso_files[:307]),
        indices=tuple(indices),
        termos_por_posicao_verso=terms,
        posicoes_rejeitadas=rejected,
    )


class OrganizedBookImporter:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    @staticmethod
    def _quality(image) -> dict:
        result = avaliar_qualidade(image)
        return {
            "qualidade_foco": float(result["foco_valor"]),
            "qualidade_exposicao": float(result["exposicao_valor"]),
            "qualidade_enquadramento": result["enquadramento_status"],
            "qualidade_oclusao": float(result["oclusao_valor"]),
            "qualidade_status": result["status_geral"],
            "qualidade_motivos": json.dumps(result["motivos_refazer"], ensure_ascii=False),
            "precisa_revisao": 1 if result["repetir_captura"] else 0,
        }

    def _insert_image(
        self,
        *,
        livro_id: int,
        path: Path,
        ordem: int,
        tipo_documento: str,
        origem_posicao: int,
        face: str = "indeterminado",
        folha: int | None = None,
        termos: tuple[int, int] | None = None,
        rotacao: int = 0,
        duplicate_status: str = "unico",
        duplicate_ref: int | None = None,
        quality: bool = False,
    ) -> int:
        image = _read_image(path)
        if image is None:
            raise ValueError(f"Imagem inválida: {path}")
        phash, dhash = compute_hashes(image)
        fields = self._quality(image) if quality else {
            "qualidade_status": "nao_aplicavel",
            "precisa_revisao": 0,
        }
        term_start, term_end = termos or (None, None)
        image_id = self.repo.registrar_imagem(
            livro_id=livro_id,
            ordem_captura=ordem,
            caminho_original=str(path),
            caminho_thumb=str(path),
            tipo_documento=tipo_documento,
            rotacao_visualizacao=rotacao,
            origem_posicao=origem_posicao,
            hash_perceptual=phash,
            dhash=dhash,
            sha256=compute_sha256(path),
            folha_estimada=folha,
            face=face,
            folha_status="confirmado_auditoria" if folha else "nao_aplicavel",
            termo_inicial=term_start,
            termo_final=term_end,
            termo_final_decidido=term_end,
            folha_final_decidida=folha,
            termo_status="confirmado_auditoria" if termos else "nao_aplicavel",
            motor_utilizado="sequencia_auditada" if termos else "classificador_documento",
            confianca_termo=1.0 if termos else None,
            confianca_folha=1.0 if folha else None,
            duplicidade_status=duplicate_status,
            duplicidade_confianca=1.0 if duplicate_ref else None,
            duplicidade_ref=duplicate_ref,
            status="aceita" if tipo_documento == "registro" and duplicate_status == "unico" else tipo_documento,
            **fields,
        )
        if tipo_documento == "registro" and duplicate_status == "unico":
            registros = self.repo.sincronizar_registros_imagem(image_id)
            for registro in registros:
                if registro.get("termo") is not None:
                    self.repo.salvar_metadado_tratado(
                        imagem_id=image_id,
                        registro_id=registro["id"],
                        tipo="termo",
                        valor=str(registro["termo"]),
                        confianca=1.0,
                        fonte="sequencia_auditada",
                        motor="auditoria_livro",
                        status="confirmado",
                        contexto="Termo validado pela sequência física do Livro A-07",
                    )
            if fields.get("precisa_revisao"):
                self.repo.criar_revisao(
                    imagem_id=image_id,
                    tipo="refazer_captura",
                    detalhes=fields.get("qualidade_motivos") or "Falha de qualidade",
                )
        return image_id

    def importar_a07(
        self,
        root: str | Path,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        audit = auditar_a07(root)
        existing = self.repo.db.fetchone(
            "SELECT id, observacoes FROM livro WHERE acervo_id=6 AND codigo='A-07' ORDER BY id LIMIT 1"
        )
        if existing:
            total = self.repo.get_total_imagens_livro(existing["id"])
            if total == 637:
                return {"livro_id": existing["id"], "ja_existia": True, "total_imagens": total}
            if total and "Importado de pasta organizada" not in (existing.get("observacoes") or ""):
                raise ValueError(
                    "Já existe um Livro A-07 não criado por este importador; a operação foi cancelada."
                )
            # Recuperacao idempotente: remove somente a importacao parcial
            # identificada pelo marcador acima. Nenhuma foto original e tocada.
            livro_id = existing["id"]
            self.repo.db.update(
                "DELETE FROM ocr_deteccao WHERE imagem_id IN (SELECT id FROM imagem WHERE livro_id=?)",
                (livro_id,),
            )
            self.repo.db.update(
                "DELETE FROM ocr_execucao WHERE imagem_id IN (SELECT id FROM imagem WHERE livro_id=?)",
                (livro_id,),
            )
            self.repo.db.update(
                "DELETE FROM revisao WHERE imagem_id IN (SELECT id FROM imagem WHERE livro_id=?)",
                (livro_id,),
            )
            self.repo.db.update("DELETE FROM registro WHERE livro_id=?", (livro_id,))
            self.repo.db.update("DELETE FROM ocorrencia WHERE livro_id=?", (livro_id,))
            self.repo.db.update("DELETE FROM imagem WHERE livro_id=?", (livro_id,))
        else:
            livro_id = self.repo.criar_livro(
                acervo_id=6,
                oficio_id=6,
                tipo_id=1,
                codigo="A-07",
                nome_capa="Livro A nº 07 — Nascimentos — 1ª Zona de Aracaju",
                total_folhas=300,
                primeira_folha=1,
                ultima_folha=300,
                frente_verso=1,
                registros_por_face=2,
                termo_inicial=6801,
                termo_final=8000,
                observacoes=(
                    "Importado de pasta organizada. Abertura e índices separados dos registros. "
                    "Auditoria detectou 10 recapturas e 3 versos ausentes."
                ),
                status="precisa_complementacao",
            )

        total_work = 1 + len(audit.frentes) + len(audit.versos) + len(audit.indices)
        done = 0

        def progress(label: str) -> None:
            nonlocal done
            done += 1
            if on_progress:
                on_progress(done, total_work, label)

        self._insert_image(
            livro_id=livro_id,
            path=audit.capa,
            ordem=0,
            tipo_documento="abertura",
            origem_posicao=1,
        )
        progress("Abertura classificada")

        for position, path in enumerate(audit.frentes, 1):
            term_start = 6801 + (position - 1) * 4
            self._insert_image(
                livro_id=livro_id,
                path=path,
                ordem=position * 2 - 1,
                tipo_documento="registro",
                origem_posicao=position,
                face="frente",
                folha=position,
                termos=(term_start, term_start + 1),
                quality=True,
            )
            progress(f"Folha {position} frente")

        group_by_position = {
            position: group
            for group in A07_DUPLICATE_GROUPS
            for position in group
        }
        accepted_ids: dict[tuple[int, ...], int] = {}
        for position, path in enumerate(audit.versos, 1):
            if position in audit.posicoes_rejeitadas:
                progress(f"Recaptura {position} separada")
                continue
            terms = audit.termos_por_posicao_verso[position]
            folha = ((terms[0] - 6803) // 4) + 1
            image_id = self._insert_image(
                livro_id=livro_id,
                path=path,
                ordem=folha * 2,
                tipo_documento="registro",
                origem_posicao=position,
                face="verso",
                folha=folha,
                termos=terms,
                rotacao=180 if position == 1 else 0,
                quality=True,
            )
            group = group_by_position.get(position)
            if group:
                accepted_ids[group] = image_id
            progress(f"Folha {folha} verso")

        for position in sorted(audit.posicoes_rejeitadas):
            path = audit.versos[position - 1]
            group = group_by_position[position]
            terms = audit.termos_por_posicao_verso[position]
            folha = ((terms[0] - 6803) // 4) + 1
            self._insert_image(
                livro_id=livro_id,
                path=path,
                ordem=10_000 + position,
                tipo_documento="recaptura_rejeitada",
                origem_posicao=position,
                face="verso",
                folha=folha,
                termos=terms,
                duplicate_status="duplicata_confirmada",
                duplicate_ref=accepted_ids[group],
                quality=True,
            )

        for position, path in enumerate(audit.indices, 1):
            self._insert_image(
                livro_id=livro_id,
                path=path,
                ordem=20_000 + position,
                tipo_documento="indice",
                origem_posicao=position,
            )
            progress(f"Índice {position}")

        for folha, term_start, term_end in A07_MISSING:
            self.repo.criar_ocorrencia(
                livro_id=livro_id,
                tipo="refazer_captura",
                folha_afetada=folha,
                termo_afetado=term_start,
                descricao=(
                    f"Foto ausente: folha {folha} verso, termos {term_start}–{term_end}. "
                    "Solicitar nova captura; a contagem não deve avançar até a foto ser aceita."
                ),
                confirmada=1,
            )

        self.repo.atualizar_livro(livro_id, status="precisa_complementacao")
        stats = self.repo.db.fetchone(
            """
            SELECT COUNT(*) total,
                   SUM(tipo_documento='registro' AND duplicidade_status='unico') uteis,
                   SUM(tipo_documento='recaptura_rejeitada') recapturas,
                   SUM(tipo_documento='indice') indices,
                   SUM(tipo_documento='abertura') aberturas,
                   SUM(precisa_revisao=1 AND tipo_documento='registro' AND duplicidade_status='unico') revisoes
            FROM imagem WHERE livro_id=?
            """,
            (livro_id,),
        ) or {}
        return {
            "livro_id": livro_id,
            "ja_existia": False,
            "total_imagens": int(stats.get("total") or 0),
            "faces_uteis": int(stats.get("uteis") or 0),
            "recapturas": int(stats.get("recapturas") or 0),
            "indices": int(stats.get("indices") or 0),
            "aberturas": int(stats.get("aberturas") or 0),
            "revisoes_qualidade": int(stats.get("revisoes") or 0),
            "faltantes": [
                {"folha": folha, "face": "verso", "termo_inicial": start, "termo_final": end}
                for folha, start, end in A07_MISSING
            ],
        }
