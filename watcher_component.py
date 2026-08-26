from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal


def arquivo_estavel(
    path: str | Path,
    timeout_seg: float = 5.0,
    intervalo_seg: float = 0.2,
    leituras_iguais: int = 2,
) -> bool:
    """Aguarda ate que o tamanho do arquivo pare de mudar.

    Retorna True quando o arquivo mantem o mesmo tamanho por
    ``leituras_iguais`` leituras consecutivas dentro do ``timeout_seg``.
    """
    p = Path(path)
    inicio = time.time()
    tamanhos_iguais = 0
    ultimo_tamanho = None
    while time.time() - inicio < timeout_seg:
        try:
            if not p.exists():
                tamanhos_iguais = 0
                ultimo_tamanho = None
            else:
                tamanho = p.stat().st_size
                if ultimo_tamanho is not None and tamanho == ultimo_tamanho:
                    tamanhos_iguais += 1
                else:
                    tamanhos_iguais = 0
                ultimo_tamanho = tamanho
                if tamanhos_iguais >= leituras_iguais:
                    return True
        except (OSError, FileNotFoundError):
            tamanhos_iguais = 0
            ultimo_tamanho = None
        time.sleep(intervalo_seg)
    return False


def carregar_pasta(settings: Any, section: str, key: str) -> Path:
    """Retorna a pasta monitorada configurada em ``section.key``."""
    valor = settings.get(section, key, "")
    return Path(str(valor))


__all__ = ["arquivo_estavel", "carregar_pasta"]
