"""Telemetria de desempenho — eventos em JSONL, sem alterar o comportamento medido.

Invariante 13: telemetria nunca altera o que mede.

- desligável via ``telemetry.enabled``;
- sem imagem no log;
- sem OCR/texto sensível/dados pessoais: apenas identificadores técnicos
  (ids, durações, contagens, flags) em uma allowlist;
- sem log por frame: eventos de transição + amostras periódicas (~1 Hz);
- sem I/O síncrono pesado no caminho crítico: uma fila `queue.Queue` recebe os
  eventos e uma thread própria escreve o JSONL.

Uso típico:
    from ..services.telemetry import emitir, configurar, iniciar_amostrador
    configurar(settings)
    iniciar_amostrador()
    emitir("ocr.fast_finished", job_id=..., duration_ms=..., uncertain=True)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Campos permitidos nos eventos. Qualquer outro é descartado (invariante 13).
CAMPOS_LIVRES = {
    "job_id", "registro_id", "imagem_id", "lote_id", "livro_id",
    "duration_ms", "queue_depth", "uncertain", "sucesso", "etapa", "motor",
    "status", "de", "para", "duracao_estado_ms", "contagem",
    "fps_preview", "fps_analisado", "detector_ms_max", "save_ms_max",
    "ui_jitter_ms_max", "fila_ocr", "fila_captura", "cpu_percent", "mem_rss_mb",
    "bytes", "qualidade", "duplicidade", "campos", "tentativas", "erro_tipo",
    "pendentes", "processando", "concluidos", "falhas", "capturas",
}

# Campos explicitamente proibidos (PII/conteúdo) — nunca emitir.
CAMPOS_PROIBIDOS = {
    "nome", "mae", "pai", "data", "texto", "texto_ocr", "cpf", "rg", "cnh",
    "imagem", "path", "caminho", "arquivo", "valor", "conteudo",
}

_travamento = threading.Lock()
_configurado = False
_ativo = False
_caminho: Path | None = None
_fila: "queue.Queue[str]" | None = None
_escritor_thread: threading.Thread | None = None
_amostradores: list[tuple[str, Callable[[], dict[str, Any]]]] = []
_sampler_thread: threading.Thread | None = None
_sampler_intervalo = 1.0
_descartados = 0

_SENTINELA = object()


def _sanitizar(campos: dict[str, Any]) -> dict[str, Any]:
    """Mantém apenas a allowlist e nunca campos proibidos."""
    global _descartados
    saida: dict[str, Any] = {}
    for chave, valor in campos.items():
        chave = str(chave).lower()
        if chave in CAMPOS_PROIBIDOS:
            _descartados += 1
            continue
        if chave not in CAMPOS_LIVRES:
            _descartados += 1
            continue
        saida[chave] = valor
    return saida


def _loop_escritor(fila: "queue.Queue[str]") -> None:
    """Escreve os eventos em thread própria: o I/O nunca toca o caminho crítico.

    A fila é recebida por argumento (nunca lida do estado global) para que
    ``parar()`` possa encerrar o ciclo sem janelas de corrida.
    """
    stream = None
    try:
        while True:
            item = fila.get()
            if item is _SENTINELA:
                break
            if stream is None:
                _caminho.parent.mkdir(parents=True, exist_ok=True)
                stream = open(_caminho, "a", encoding="utf-8", newline="\n")
            stream.write(item + "\n")
            stream.flush()
    except Exception:
        pass
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def configurar(settings, caminho: str | Path | None = None) -> None:
    """Configura o destino dos eventos. Idempotente; pode ser chamado a qualquer hora."""
    global _configurado, _ativo, _caminho, _fila, _escritor_thread
    with _travamento:
        ativo = bool(settings.get("telemetry", "enabled", True))
        base = Path(str(caminho or "") or str(settings.get("telemetry", "path", "") or ""))
        if not base.is_absolute():
            from ..config.settings import data_dir
            base = data_dir() / (base or Path("telemetry"))
        destino = base / "events.jsonl"
        if not ativo:
            _ativo = False
            return
        if _configurado and _ativo and _caminho == destino:
            return
        nova_fila: "queue.Queue[str]" = queue.Queue()
        _fila = nova_fila
        _escritor_thread = threading.Thread(
            target=_loop_escritor, args=(nova_fila,),
            name="telemetry-writer", daemon=True,
        )
        _escritor_thread.start()
        _caminho = destino
        _ativo = True
        _configurado = True
        logger.info("Telemetria ativa em %s", destino)


def emitir(evento: str, **campos: Any) -> None:
    """Publica um evento (no-op se desligado). Nunca bloqueia o chamador por I/O."""
    if not _ativo or _fila is None:
        return
    linha = {
        "event": evento,
        "ts": round(time.time(), 3),
        **_sanitizar(campos),
    }
    try:
        _fila.put_nowait(json.dumps(linha, ensure_ascii=False))
    except Exception:
        pass


def registrar_amostrador(nome: str, funcao: Callable[[], dict[str, Any]] | None) -> None:
    """Registra (ou remove, com ``funcao=None``) um coletor chamado a ~1 Hz."""
    global _amostradores
    with _travamento:
        _amostradores = [item for item in _amostradores if item[0] != nome]
        if funcao is not None:
            _amostradores.append((nome, funcao))


def _amostrar() -> None:
    try:
        import psutil
        processo = psutil.Process()
        campos: dict[str, Any] = {
            "cpu_percent": round(processo.cpu_percent(interval=None), 1),
            "mem_rss_mb": round(processo.memory_info().rss / 1024 / 1024, 1),
        }
        with _travamento:
            amostradores = list(_amostradores)
        for _nome, funcao in amostradores:
            try:
                campos.update(funcao())
            except Exception:
                pass
        emitir("resource.sample", **campos)
    except Exception:
        pass


def _loop_amostrador() -> None:
    proxima = time.monotonic() + _sampler_intervalo
    while _ativo:
        agora = time.monotonic()
        if agora >= proxima:
            _amostrar()
            proxima = agora + _sampler_intervalo
        time.sleep(max(0.05, min(0.25, _sampler_intervalo / 4)))


def iniciar_amostrador(intervalo: float = 1.0) -> None:
    """Inicia a amostragem periódica de CPU/RAM e coletores registrados."""
    global _sampler_thread, _sampler_intervalo
    if not _ativo:
        return
    _sampler_intervalo = max(0.5, float(intervalo))
    with _travamento:
        if _sampler_thread is not None and _sampler_thread.is_alive():
            return
        _sampler_thread = threading.Thread(
            target=_loop_amostrador, name="telemetry-sampler", daemon=True
        )
        _sampler_thread.start()


def parar() -> None:
    """Encerra a amostragem e esvazia a fila de eventos (usar no fim do app/teste)."""
    global _ativo, _fila, _escritor_thread
    with _travamento:
        _ativo = False
        fila = _fila
        _fila = None
        escritor = _escritor_thread
        _escritor_thread = None
    if fila is not None:
        try:
            fila.put_nowait(_SENTINELA)
        except Exception:
            pass
    if escritor is not None and escritor.is_alive():
        escritor.join(timeout=2.0)
