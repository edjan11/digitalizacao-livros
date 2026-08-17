from __future__ import annotations

"""Fila persistente de nomes: OCR rápido em paralelo e Qwen sequencial."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import unicodedata
from typing import Callable

import cv2
import numpy as np

from ..database.connection import Database
from ..database.repository import Repository
from ..imaging.document import RectificationResult, retificar_formulario
from ..imaging.record_regions import (
    bbox_data_registro,
    bbox_corresponde_registro,
    bbox_faixa_nome,
    bbox_linha_nome,
    bbox_registro,
    recortar_bbox,
)
from ..imaging.book_layouts import name_bbox_for_book, record_bbox_for_book
from ..metadata.extractor import extrair_metadados
from ..metadata.normalizer import normalizar_busca, tratar_valor
from ..ocr.engines import RapidOCRProvider, TesseractProvider
from ..ocr.qwen_vl_engine import QwenRecordAnalyzer, modelo_qwen_instalado
from .telemetry import emitir, registrar_amostrador


ProgressCallback = Callable[[dict, str], None]
_thread_local = threading.local()


@contextmanager
def _exclusive_worker_lock(db_path: Path):
    """Impede dois processos Qwen de consumirem o mesmo item simultaneamente."""
    lock_path = Path(db_path).with_suffix(Path(db_path).suffix + ".nomes.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("r+b") if lock_path.exists() else lock_path.open("w+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            raise RuntimeError(
                "ja existe um trabalhador de nomes ativo para este banco"
            ) from exc
        stream.seek(0)
        stream.write(str(os.getpid()).encode("ascii", "ignore").ljust(20, b" "))
        stream.truncate(20)
        stream.flush()
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _bbox_na_geometria(bbox, geometria):
    x1, y1, x2, y2 = bbox
    if geometria.left_line:
        x1 = max(x1, sum(geometria.left_line) / 2 + 0.003)
    if geometria.right_line:
        x2 = min(x2, sum(geometria.right_line) / 2 - 0.003)
    return (x1, y1, x2, y2)


def _abrir_imagem(item: dict) -> tuple[np.ndarray, str]:
    armazenamento = Path(item.get("caminho_armazenamento") or "")
    usa_armazenamento = armazenamento.is_file()
    caminho = armazenamento if usa_armazenamento else Path(item.get("caminho_original") or "")
    if not caminho.is_file():
        raise RuntimeError("fotografia não encontrada")
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        raise RuntimeError("fotografia vazia")
    sha = hashlib.sha256(dados.tobytes()).hexdigest()
    esperado = str(
        item.get("sha256_armazenamento")
        if usa_armazenamento
        else (item.get("sha256_atual") or item.get("imagem_sha256") or "")
    )
    if esperado and sha.lower() != esperado.lower():
        raise RuntimeError("hash da fotografia mudou; recaptura precisa ser sincronizada")
    image = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("não foi possível abrir a fotografia")
    normalized = Path(item.get("caminho_normalizado") or "")
    uses_normalized = normalized.is_file() and armazenamento == normalized
    if not uses_normalized:
        rotacao = int(item.get("rotacao_visualizacao") or 0)
        if rotacao == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotacao == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotacao == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image, sha


def _geometria_item(image: np.ndarray, item: dict) -> RectificationResult:
    normalized = Path(item.get("caminho_normalizado") or "")
    storage = Path(item.get("caminho_armazenamento") or "")
    if normalized.is_file() and storage == normalized:
        return RectificationResult(
            image=image,
            applied=False,
            confidence=1.0,
            angle_degrees=0.0,
            reason="derivada normalizada; sem segunda transformacao",
        )
    return retificar_formulario(image)


def _provedores(settings) -> list:
    providers = getattr(_thread_local, "providers", None)
    if providers is None:
        providers = []
        tesseract = TesseractProvider(
            tesseract_path=settings.get("ocr", "tesseract_path", "") or None,
            lang=settings.get("ocr", "lang", "por"),
        )
        rapid = RapidOCRProvider(apenas_cabecalhos=False)
        if tesseract.is_available():
            providers.append(tesseract)
        if rapid.is_available():
            providers.append(rapid)
        _thread_local.providers = providers
    return providers


def _localizar_linha_nome(image: np.ndarray, search_bbox, providers: list, *, required: bool):
    """Narrow a book-specific search band using the printed name label."""
    search_crop = recortar_bbox(image, search_bbox)
    rapid = next((provider for provider in providers if provider.name == "rapidocr"), None)
    if rapid is None or search_crop is None or search_crop.size == 0:
        if required:
            raise RuntimeError("RapidOCR indisponivel para localizar a linha do nome")
        return search_bbox, "fallback_sem_rapidocr"
    result = rapid.recognize(search_crop, fast=True)
    sx1, sy1, sx2, sy2 = (float(value) for value in search_bbox)
    sw, sh = sx2 - sx1, sy2 - sy1
    candidates = []
    minute_lines = []
    for token in result.tokens:
        box = getattr(token, "bbox", None)
        text = unicodedata.normalize("NFKD", str(getattr(token, "valor", "")))
        text = text.encode("ascii", "ignore").decode().lower()
        text = re.sub(r"[^a-z]+", " ", text)
        if not box:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        words = set(text.split())
        if "minutos" in words:
            minute_lines.append((min(ys), max(ys)))
        # `nome` and `de` are frequently returned as separate boxes.  Whole
        # words avoid confusing `nomeadas` in the witness sentence with the
        # target label.
        if "recebeu" in text or "nome" in words:
            candidates.append((float(getattr(token, "confianca", 0)), min(xs), max(xs), min(ys), max(ys)))
    if not candidates:
        if minute_lines:
            # In this fixed form the sex/name lines immediately follow the
            # hour/minute line.  This fallback remains inside the calibrated
            # search band and is safer than passing the entire record.
            _min_y, max_y = max(minute_lines, key=lambda value: value[1])
            line = (
                sx1, sy1 + sh * min(0.82, max_y + 0.12), sx2,
                sy1 + sh * min(0.99, max_y + 0.39),
            )
            return line, "rapidocr_ancora_minutos"
        if required:
            raise RuntimeError("rotulo 'que recebeu o nome de' nao localizado no gabarito A-16")
        return search_bbox, "fallback_sem_rotulo"
    _confidence, _min_x, max_x, min_y, max_y = max(candidates)
    # Start immediately after the printed label.  A small vertical allowance
    # preserves ascenders/descenders but cannot reach sex or filiation lines.
    line = (
        min(sx2 - 0.02, max(sx1, sx1 + sw * max_x + 0.004)),
        max(sy1, sy1 + sh * min_y - 0.010),
        sx2,
        min(sy2, sy1 + sh * max_y + 0.014),
    )
    if line[2] - line[0] < 0.20 or line[3] - line[1] < 0.012:
        if required:
            raise RuntimeError("linha do nome localizada com geometria invalida")
        return search_bbox, "fallback_geometria_invalida"
    return line, "rapidocr_rotulo_posicional"


def _localizar_linhas_nome_pagina(
    image: np.ndarray, providers: list, *, total: int
) -> dict[int, tuple[tuple[float, float, float, float], str]]:
    """Localiza todos os campos de nome numa leitura da face inteira.

    A coordenada vertical fixa falha quando a câmera deixa mais ou menos margem
    no topo. O rótulo impresso é estável e serve somente como âncora; o Qwen
    continua recebendo os pixels originais do manuscrito ao lado do rótulo.
    """
    rapid = next((provider for provider in providers if provider.name == "rapidocr"), None)
    if rapid is None:
        return {}
    search_bbox = (0.05, 0.015, 0.74, 0.985)
    result = rapid.recognize(recortar_bbox(image, search_bbox), fast=True)
    sx1, sy1, sx2, sy2 = search_bbox
    sw, sh = sx2 - sx1, sy2 - sy1
    candidates = []
    for token in result.tokens:
        box = getattr(token, "bbox", None)
        if not box:
            continue
        normalized = unicodedata.normalize("NFKD", str(getattr(token, "valor", "")))
        normalized = normalized.encode("ascii", "ignore").decode().lower()
        # Em fotos reais o RapidOCR leu "recebeuomme" em uma das duas
        # linhas. "recebeu" é distintivo neste formulário e não se confunde
        # com a frase de testemunhas que contém apenas "nomeadas".
        if "recebeu" not in re.sub(r"[^a-z]+", "", normalized):
            continue
        confidence = float(getattr(token, "confianca", 0) or 0)
        if confidence < 0.70:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        bbox = (
            # A escrita frequentemente invade o fim do rótulo impresso. O X
            # calibrado preserva a primeira letra; a âncora dinâmica decide Y.
            0.28,
            max(0.0, sy1 + sh * min(ys) - 0.010),
            sx2,
            min(1.0, sy1 + sh * max(ys) + 0.018),
        )
        candidates.append(((bbox[1] + bbox[3]) / 2, confidence, bbox))
    deduplicated = []
    for candidate in sorted(candidates, key=lambda value: (-value[1], value[0])):
        if any(abs(candidate[0] - kept[0]) < 0.035 for kept in deduplicated):
            continue
        deduplicated.append(candidate)
    deduplicated.sort(key=lambda value: value[0])
    if total == 2 and len(deduplicated) == 1:
        center, confidence, bbox = deduplicated[0]
        shift = -0.50 if center >= 0.50 else 0.50
        inferred = (
            bbox[0], max(0.0, bbox[1] + shift), bbox[2], min(1.0, bbox[3] + shift)
        )
        inferred_candidate = (
            (inferred[1] + inferred[3]) / 2,
            max(0.70, confidence * 0.80),
            inferred,
        )
        deduplicated.append(inferred_candidate)
        deduplicated.sort(key=lambda value: value[0])
    if len(deduplicated) > total:
        selected = []
        remaining = list(deduplicated)
        for index in range(total):
            target = (index + 0.5) / total
            best = min(remaining, key=lambda value: abs(value[0] - target))
            selected.append(best)
            remaining.remove(best)
        deduplicated = sorted(selected, key=lambda value: value[0])
    if len(deduplicated) != total:
        return {}
    return {
        index: (bbox, f"rapidocr_pagina_rotulo_{confidence:.2f}")
        for index, (_center, confidence, bbox) in enumerate(deduplicated)
    }


def _candidatos_nome(resultado) -> list[dict]:
    candidatos = [
        {
            "valor": str(d.valor_tratado).strip(),
            "confianca": float(d.confianca),
            "motor": d.motor or resultado.motor,
            "contexto": d.contexto or "",
        }
        for d in extrair_metadados(resultado)
        if d.tipo == "nome_registrado" and str(d.valor_tratado).strip()
    ]
    # A confiança fixa do extrator (`nome de ...` = 0,72) não é uma
    # confiança de leitura. Para Tesseract, que não devolve uma probabilidade
    # calibrada por palavra, mantemos um teto conservador. RapidOCR pode
    # fornecer confiança própria nos tokens, mas continua dependendo de uma
    # segunda evidência para virar sugestão.
    if (resultado.motor or "").lower().startswith("tesseract"):
        for candidato in candidatos:
            candidato["confianca"] = min(candidato["confianca"], 0.48)
    return candidatos


def _nome_parece_legivel(valor: str) -> bool:
    """Filtro de triagem; não tenta decidir a grafia correta.

    O objetivo é impedir que ruído com números, pontuação ou palavras
    quebradas seja apresentado como nome verde/neutro. A confirmação real
    continua sendo do operador ou do Qwen.
    """
    texto = str(valor or "").strip()
    if not texto or any(ch.isdigit() for ch in texto):
        return False
    if re.search(r"[^A-Za-zÀ-ÿ\s'’\-]", texto):
        return False
    if re.search(r"\s[-–—]\s", texto):
        return False
    palavras = re.findall(r"[A-Za-zÀ-ÿ]+", texto)
    if not 2 <= len(palavras) <= 12:
        return False
    if sum(len(palavra) for palavra in palavras) < 6:
        return False
    if len("".join(palavras)) / max(1, len(texto)) < 0.72:
        return False
    curtas_permitidas = {"a", "e", "da", "das", "de", "do", "dos"}
    if any(len(palavra) == 1 and palavra.lower() not in curtas_permitidas for palavra in palavras):
        return False
    normalizado = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    if re.search(r"(.)\1\1", normalizado):
        return False
    respostas_sem_nome = (
        "nao consigo", "nao e possivel", "nao foi possivel", "nome ilegivel",
        "texto ilegivel", "imagem ilegivel", "nao legivel", "que recebeu o nome",
        "nao ha nome", "sem nome",
    )
    if any(frase in normalizado for frase in respostas_sem_nome):
        return False
    palavras_nao_nome = {
        "aracaju", "sergipe", "cartorio", "clinica", "maternidade",
        "horas", "minutos", "numero", "nascimento",
    }
    if set(normalizado.split()) & palavras_nao_nome:
        return False
    return True


def _livros_qwen_direto(settings) -> set[str]:
    value = settings.get("ocr", "name_direct_qwen_books", ["A-07", "A-16"])
    if isinstance(value, str):
        value = re.split(r"[,;]", value)
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {
        str(item).strip().upper().replace("_", "-")
        for item in value
        if str(item).strip()
    }


class NameBatchRunner:
    def __init__(
        self,
        *,
        db_path: Path,
        settings,
        lote_id: int,
        max_workers: int = 4,
        max_qwen_items: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.settings = settings
        self.lote_id = int(lote_id)
        self.max_workers = max(1, min(int(max_workers), 4))
        self.max_qwen_items = (
            None if max_qwen_items is None else max(1, int(max_qwen_items))
        )
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _summary, _label: None)
        self._qwen_bbox_cache: dict[
            tuple[str, int, int],
            dict[int, tuple[tuple[float, float, float, float], str]],
        ] = {}

    def _emitir(self, repo: Repository, label: str) -> None:
        self.on_progress(repo.resumo_processamento(self.lote_id), label)

    def _processar_rapido(self, repo: Repository, item: dict) -> None:
        inicio = time.perf_counter()
        bbox = name_bbox_for_book(
            item.get("livro_codigo"),
            int(item.get("indice_na_imagem") or 0),
            int(item.get("total_na_imagem") or 1),
        )
        emitir("ocr.fast_started", job_id=item.get("id"), registro_id=item.get("registro_id"),
               imagem_id=item.get("imagem_id"), lote_id=self.lote_id, etapa="ocr_fast")
        repo.atualizar_item_processamento(
            int(item["id"]),
            status="processando",
            tentativas=int(item.get("tentativas") or 0) + 1,
            iniciado_em=__import__("datetime").datetime.now().isoformat(),
            bbox_json=json.dumps(bbox),
            erro=None,
        )
        image, sha = _abrir_imagem(item)
        geometria = _geometria_item(image, item)
        image = geometria.image
        bbox = _bbox_na_geometria(bbox, geometria)
        repo.atualizar_item_processamento(int(item["id"]), bbox_json=json.dumps(bbox))
        providers = _provedores(self.settings)
        bbox, locator_method = _localizar_linha_nome(
            image, bbox, providers, required=str(item.get("livro_codigo") or "").upper() == "A-16"
        )
        repo.atualizar_item_processamento(int(item["id"]), bbox_json=json.dumps(bbox))
        faixa = recortar_bbox(image, bbox)
        if faixa is None or faixa.size == 0 or faixa.shape[0] < 12:
            raise RuntimeError("faixa do nome ficou pequena ou vazia")
        if faixa.shape[0] < 64:
            escala = 64 / faixa.shape[0]
            faixa = cv2.resize(
                faixa,
                (max(1, int(faixa.shape[1] * escala)), 64),
                interpolation=cv2.INTER_CUBIC,
            )

        candidatos: list[dict] = []
        textos: list[str] = []
        reconheceu_texto = False
        motores: list[str] = []
        for provider in providers:
            resultado = provider.recognize(faixa, fast=True)
            texto = (resultado.texto_bruto or "").strip()
            reconheceu_texto = reconheceu_texto or bool(texto)
            textos.append(f"[{provider.name}]\n{texto}")
            motores.append(provider.name)
            execucao = repo.criar_execucao_ocr(
                imagem_id=int(item["imagem_id"]),
                registro_id=int(item["registro_id"]),
                motor=f"{provider.name}-nome-faixa-v2",
                texto_bruto=texto,
                tempo_ms=float(resultado.tempo_ms or 0),
                sucesso=bool(texto or resultado.tokens),
                erro="" if texto or resultado.tokens else "nenhum texto reconhecido",
            )
            # Tokens e texto bruto ficam na execução; apenas um candidato de
            # nome será promovido como metadado pesquisável.
            _ = execucao
            candidatos.extend(_candidatos_nome(resultado))

        if not motores:
            raise RuntimeError("Tesseract e RapidOCR indisponíveis")

        agrupados: dict[str, dict] = {}
        for candidato in candidatos:
            chave = normalizar_busca(candidato["valor"])
            if not chave:
                continue
            anterior = agrupados.get(chave)
            if anterior is None or candidato["confianca"] > anterior["confianca"]:
                agrupados[chave] = candidato
        ordenados = sorted(
            agrupados.values(), key=lambda candidato: candidato["confianca"], reverse=True
        )
        melhor = ordenados[0] if ordenados else None
        confianca = 0.0
        concordantes = 0
        motores_concordantes: set[str] = set()
        status = "revisar" if reconheceu_texto else "sem_resultado"
        resultado_nome = ""
        if melhor:
            resultado_nome = tratar_valor(melhor["valor"])
            chave_melhor = normalizar_busca(resultado_nome)
            correspondentes = [
                candidato for candidato in candidatos
                if normalizar_busca(candidato["valor"]) == chave_melhor
            ]
            motores_concordantes = {
                str(candidato.get("motor") or "").lower()
                for candidato in correspondentes
                if candidato.get("motor")
            }
            concordantes = len(motores_concordantes)
            confianca = min(
                0.95,
                float(melhor["confianca"]) + (0.18 if concordantes >= 2 else 0.0),
            )
            legivel = _nome_parece_legivel(resultado_nome)
            status_deteccao = (
                "sugestao"
                if (
                    concordantes >= 2
                    and confianca >= 0.80
                    and geometria.confidence >= 0.70
                    and legivel
                )
                else "precisa_revisao"
            )
            status = "sugestao" if status_deteccao == "sugestao" else "revisar"
            execucao = repo.criar_execucao_ocr(
                imagem_id=int(item["imagem_id"]),
                registro_id=int(item["registro_id"]),
                motor="ocr-nomes-rapido-v2",
                texto_bruto="\n\n".join(textos),
                tempo_ms=(time.perf_counter() - inicio) * 1000,
                sucesso=True,
            )
            repo.salvar_deteccoes_ocr(
                execucao_id=execucao,
                imagem_id=int(item["imagem_id"]),
                registro_id=int(item["registro_id"]),
                deteccoes=[{
                    "tipo": "nome_registrado",
                    "valor_original": resultado_nome,
                    "valor_tratado": resultado_nome,
                    "valor_normalizado": normalizar_busca(resultado_nome),
                    "confianca": confianca,
                    "motor": melhor.get("motor") or "ocr-rapido",
                    "fonte": "ocr_nome_rapido",
                    "status": status_deteccao,
                    "bbox_json": json.dumps(bbox),
                    "contexto": f"concordancia_motores={concordantes}; " + (
                        f"Faixa do nome; concordância={concordantes}; "
                        f"geometria={geometria.confidence:.2f}; {geometria.reason}; "
                        f"localizador={locator_method}"
                    ),
                }],
            )
        else:
            repo.criar_execucao_ocr(
                imagem_id=int(item["imagem_id"]),
                registro_id=int(item["registro_id"]),
                motor="ocr-nomes-rapido-v2",
                texto_bruto="\n\n".join(textos),
                tempo_ms=(time.perf_counter() - inicio) * 1000,
                sucesso=False,
                erro="nenhum candidato de nome",
            )

        repo.atualizar_item_processamento(
            int(item["id"]),
            imagem_sha256=sha,
            status=status,
            motor="+".join(motores),
            resultado=resultado_nome,
            confianca=confianca,
            tempo_ms=(time.perf_counter() - inicio) * 1000,
            concluido_em=__import__("datetime").datetime.now().isoformat(),
            erro=None,
        )
        emitir("ocr.fast_finished", job_id=item.get("id"), registro_id=item.get("registro_id"),
               lote_id=self.lote_id, duration_ms=round((time.perf_counter() - inicio) * 1000),
               uncertain=status not in ("sugestao", "encaminhado_qwen"), sucesso=status != "sem_resultado",
               motor="+".join(motores))
        limiar = float(self.settings.get("ocr", "name_qwen_threshold", 0.78))
        if status != "sugestao" or confianca < limiar:
            item["bbox_json"] = json.dumps(bbox)
            repo.garantir_item_qwen(item)

    def _processar_rapido_com_retry(self, repo: Repository, item: dict) -> None:
        erro = ""
        tentativas_iniciais = int(item.get("tentativas") or 0)
        for tentativa in range(2):
            try:
                item_tentativa = dict(item)
                item_tentativa["tentativas"] = tentativas_iniciais + tentativa
                self._processar_rapido(repo, item_tentativa)
                return
            except Exception as exc:
                erro = str(exc)
        emitir("job.failed", job_id=item.get("id"), registro_id=item.get("registro_id"),
               lote_id=self.lote_id, etapa="ocr_fast", tentativas=2, erro_tipo=erro[:200])
        repo.atualizar_item_processamento(
            int(item["id"]),
            status="falhou",
            tentativas=tentativas_iniciais + 2,
            erro=erro[:1000],
            concluido_em=__import__("datetime").datetime.now().isoformat(),
        )

    def _encaminhar_qwen_direto(self, repo: Repository, item: dict) -> None:
        """Cria o trabalho Qwen sem gastar uma leitura OCR redundante."""
        bbox = name_bbox_for_book(
            item.get("livro_codigo"),
            int(item.get("indice_na_imagem") or 0),
            int(item.get("total_na_imagem") or 1),
        )
        item_qwen = dict(item)
        item_qwen["bbox_json"] = json.dumps(bbox)
        repo.garantir_item_qwen(item_qwen)
        repo.atualizar_item_processamento(
            int(item["id"]),
            status="encaminhado_qwen",
            motor="qwen_direto_layout",
            resultado="",
            confianca=0.0,
            tempo_ms=0.0,
            bbox_json=json.dumps(bbox),
            concluido_em=__import__("datetime").datetime.now().isoformat(),
            erro=None,
        )

    def _processar_qwen(self, repo: Repository, analisador: QwenRecordAnalyzer, item: dict) -> None:
        indice = int(item.get("indice_na_imagem") or 0)
        total = int(item.get("total_na_imagem") or 1)
        bbox = name_bbox_for_book(item.get("livro_codigo"), indice, total)
        try:
            stored_bbox = json.loads(item.get("bbox_json") or "null")
            if isinstance(stored_bbox, list) and len(stored_bbox) == 4:
                bbox = tuple(float(value) for value in stored_bbox)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        assento = record_bbox_for_book(item.get("livro_codigo"), indice, total)
        if not bbox_corresponde_registro(
            (
                bbox[0],
                # A faixa é interna; para validar a associação vertical usamos
                # o assento completo guardado separadamente no contexto.
                assento[1],
                bbox[2],
                assento[3],
            ),
            indice,
            total,
        ):
            raise RuntimeError("região Qwen incompatível com o assento")
        inicio = time.perf_counter()
        repo.atualizar_item_processamento(
            int(item["id"]),
            status="processando",
            tentativas=int(item.get("tentativas") or 0) + 1,
            iniciado_em=__import__("datetime").datetime.now().isoformat(),
            bbox_json=json.dumps(bbox),
            erro=None,
        )
        image, sha = _abrir_imagem(item)
        geometria = _geometria_item(image, item)
        image = geometria.image
        locator_method = "gabarito_fixo"
        book_code = str(item.get("livro_codigo") or "").upper().replace("_", "-")
        if book_code in _livros_qwen_direto(self.settings):
            cache_key = (sha, int(item.get("rotacao_visualizacao") or 0), total)
            lines = self._qwen_bbox_cache.get(cache_key)
            if lines is None:
                lines = _localizar_linhas_nome_pagina(
                    image, _provedores(self.settings), total=total
                )
                self._qwen_bbox_cache[cache_key] = lines
            located = lines.get(indice)
            if not located:
                raise RuntimeError(
                    "rótulo 'que recebeu o nome de' não localizado; "
                    "Qwen bloqueado para evitar campo errado"
                )
            bbox, locator_method = located
        bbox = _bbox_na_geometria(bbox, geometria)
        repo.atualizar_item_processamento(int(item["id"]), bbox_json=json.dumps(bbox))
        recorte = recortar_bbox(image, bbox)
        if recorte is None or recorte.size == 0:
            raise RuntimeError("faixa Qwen do nome ficou vazia")
        emitir("qwen.started", job_id=item.get("id"), registro_id=item.get("registro_id"),
               lote_id=self.lote_id, etapa="qwen_nome")
        inicio_qwen = time.perf_counter()
        nome_bruto, resultado = analisador.analisar_nome(recorte)
        nome = tratar_valor(nome_bruto)
        # Confirma novamente o hash e a associação antes de persistir.
        atual = repo.db.fetchone(
            """
            SELECT r.imagem_id, r.indice_na_imagem,
                   COALESCE(i.sha256_armazenamento, i.sha256) AS sha256,
                   (SELECT COUNT(*) FROM registro rr WHERE rr.imagem_id=r.imagem_id) AS total
            FROM registro r JOIN imagem i ON i.id=r.imagem_id WHERE r.id=?
            """,
            (int(item["registro_id"]),),
        )
        if not atual or int(atual["imagem_id"]) != int(item["imagem_id"]):
            raise RuntimeError("registro mudou durante o processamento")
        if str(atual.get("sha256") or "").lower() not in {"", sha.lower()}:
            raise RuntimeError("fotografia mudou durante o processamento")
        if int(atual.get("indice_na_imagem") or 0) != indice or int(atual.get("total") or 1) != total:
            raise RuntimeError("posição do assento mudou durante o processamento")

        if nome and _nome_parece_legivel(nome):
            execucao = repo.criar_execucao_ocr(
                imagem_id=int(item["imagem_id"]),
                registro_id=int(item["registro_id"]),
                motor="qwen-nome-faixa-v3",
                texto_bruto=resultado.texto_bruto or nome,
                tempo_ms=float(resultado.tempo_ms or 0),
                sucesso=True,
            )
            deteccoes = [{
                    "tipo": "nome_registrado",
                    "valor_original": nome,
                    "valor_tratado": nome,
                    "valor_normalizado": normalizar_busca(nome),
                    "confianca": 0.60,
                    "motor": "qwen2-vl-2b-nome-faixa",
                    "fonte": "qwen_nome_correcao",
                    "status": "precisa_revisao",
                    "bbox_json": json.dumps(bbox),
                    "contexto": (
                        "Sugestão Qwen na faixa correta; confirmar no revisor. "
                        f"Geometria={geometria.confidence:.2f}; {geometria.reason}; "
                        f"localizador={locator_method}"
                    ),
                }]
            repo.salvar_deteccoes_ocr(
                execucao_id=execucao,
                imagem_id=int(item["imagem_id"]),
                registro_id=int(item["registro_id"]),
                deteccoes=deteccoes,
            )
            status = "revisar"
        else:
            if nome:
                repo.criar_execucao_ocr(
                    imagem_id=int(item["imagem_id"]),
                    registro_id=int(item["registro_id"]),
                    motor="qwen-nome-faixa-v3",
                    texto_bruto=resultado.texto_bruto or nome,
                    tempo_ms=float(resultado.tempo_ms or 0),
                    sucesso=False,
                    erro="resposta sem formato plausivel de nome",
                )
                nome = ""
            status = "sem_resultado"
        repo.atualizar_item_processamento(
            int(item["id"]),
            imagem_sha256=sha,
            status=status,
            motor="qwen2-vl-2b-nome-faixa",
            resultado=nome,
            confianca=0.60 if nome else 0.0,
            tempo_ms=(time.perf_counter() - inicio) * 1000,
            concluido_em=__import__("datetime").datetime.now().isoformat(),
            erro=None,
        )
        emitir("qwen.finished", job_id=item.get("id"), registro_id=item.get("registro_id"),
               lote_id=self.lote_id, duration_ms=round((time.perf_counter() - inicio_qwen) * 1000),
               sucesso=status != "sem_resultado", campos=1 if nome else 0)

    def _processar_qwen_com_retry(
        self, repo: Repository, analisador: QwenRecordAnalyzer, item: dict
    ) -> None:
        erro = ""
        tentativas_iniciais = int(item.get("tentativas") or 0)
        for tentativa in range(2):
            try:
                item_tentativa = dict(item)
                item_tentativa["tentativas"] = tentativas_iniciais + tentativa
                self._processar_qwen(repo, analisador, item_tentativa)
                return
            except Exception as exc:
                erro = str(exc)
        emitir("job.failed", job_id=item.get("id"), registro_id=item.get("registro_id"),
               lote_id=self.lote_id, etapa="qwen_nome", tentativas=2, erro_tipo=erro[:200])
        repo.atualizar_item_processamento(
            int(item["id"]),
            status="falhou",
            tentativas=tentativas_iniciais + 2,
            erro=erro[:1000],
            concluido_em=__import__("datetime").datetime.now().isoformat(),
        )

    def run(self) -> dict:
        with _exclusive_worker_lock(self.db_path):
            return self._run_locked()

    def _run_locked(self) -> dict:
        db = Database(self.db_path)
        db.connect()
        repo = Repository(db)
        try:
            repo.preparar_retomada_lote(self.lote_id)
            repo.marcar_lote_status(self.lote_id, "processando")

            def _profundidade_fila() -> dict:
                linha = repo.db.fetchone(
                    """
                    SELECT COUNT(*) AS n FROM processamento_item
                    WHERE lote_id=? AND status IN ('pendente','processando')
                    """,
                    (self.lote_id,),
                ) or {"n": 0}
                return {"queue_depth": int(linha["n"])}

            registrar_amostrador("fila_nomes", _profundidade_fila)
            pendentes = repo.listar_itens_processamento(
                self.lote_id,
                etapa="ocr_nome_rapido",
                statuses=("pendente",),
            )
            direto = _livros_qwen_direto(self.settings)
            pendentes_diretos = [
                item for item in pendentes
                if str(item.get("livro_codigo") or "").upper().replace("_", "-") in direto
            ]
            if pendentes_diretos:
                encaminhados = 0
                for item in pendentes_diretos:
                    if self.should_stop():
                        break
                    self._encaminhar_qwen_direto(repo, item)
                    encaminhados += 1
                ids_encaminhados = {
                    int(item["id"]) for item in pendentes_diretos[:encaminhados]
                }
                pendentes = [
                    item for item in pendentes if int(item["id"]) not in ids_encaminhados
                ]
                self._emitir(
                    repo,
                    f"{encaminhados} registro(s) preparados para o Qwen",
                )
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for inicio in range(0, len(pendentes), self.max_workers):
                    if self.should_stop():
                        break
                    grupo = pendentes[inicio:inicio + self.max_workers]
                    futures = {
                        executor.submit(self._processar_rapido_com_retry, repo, item): item
                        for item in grupo
                    }
                    for future in as_completed(futures):
                        future.result()
                        item = futures[future]
                        self._emitir(repo, f"OCR rápido — termo {item.get('termo') or '?'}")

            itens_qwen = repo.listar_itens_processamento(
                self.lote_id,
                etapa="qwen_nome",
                statuses=("pendente",),
                limite=self.max_qwen_items or 5000,
            )
            qwen_disponivel = modelo_qwen_instalado(
                self.settings.get("ocr", "qwen_model_path", "") or None
            )
            if not self.should_stop() and itens_qwen and qwen_disponivel:
                analisador = QwenRecordAnalyzer(
                    model_path=self.settings.get("ocr", "qwen_model_path", "") or None,
                    permitir_download=False,
                    # Quatro campos podem ocupar mais tokens que o nome isolado;
                    # aumentar o teto evita truncar o JSON no meio da data.
                    max_new_tokens=min(128, int(self.settings.get("ocr", "qwen_max_new_tokens", 128))),
                    min_pixels=int(self.settings.get("ocr", "qwen_min_pixels", 128 * 28 * 28)),
                    max_pixels=int(self.settings.get("ocr", "qwen_max_pixels", 384 * 28 * 28)),
                    dtype=str(self.settings.get("ocr", "qwen_dtype", "auto")),
                    threads=int(self.settings.get("ocr", "qwen_threads", 0)),
                )
                try:
                    for item in itens_qwen:
                        if self.should_stop():
                            break
                        self._processar_qwen_com_retry(repo, analisador, item)
                        self._emitir(repo, f"Qwen — termo {item.get('termo') or '?'}")
                finally:
                    analisador.liberar()

            restantes = repo.db.fetchone(
                """
                SELECT COUNT(*) AS n FROM processamento_item
                WHERE lote_id=? AND status IN ('pendente','processando')
                """,
                (self.lote_id,),
            ) or {"n": 0}
            status = "concluido"
            if (
                self.should_stop()
                or (itens_qwen and not qwen_disponivel)
                or int(restantes["n"] or 0) > 0
            ):
                status = "pausado"
            repo.marcar_lote_status(self.lote_id, status)
            resumo = repo.resumo_processamento(self.lote_id)
            contagens = resumo.get("contagens") or {}
            pendentes = sum(
                int(n) for chave, n in contagens.items() if chave.endswith(":pendente")
            )
            processando = sum(
                int(n) for chave, n in contagens.items() if chave.endswith(":processando")
            )
            falhas = sum(
                int(n) for chave, n in contagens.items() if chave.endswith(":falhou")
            )
            emitir("lote.status", lote_id=self.lote_id, status=status,
                   pendentes=pendentes, processando=processando,
                   concluidos=max(0, int(resumo.get("itens") or 0) - pendentes - processando - falhas),
                   falhas=falhas)
            self._emitir(repo, "Processamento pausado" if status == "pausado" else "Fila concluída")
            return resumo
        finally:
            registrar_amostrador("fila_nomes", None)
            db.close()
