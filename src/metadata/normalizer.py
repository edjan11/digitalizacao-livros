from __future__ import annotations

import re

from unidecode import unidecode


def tratar_valor(valor: str) -> str:
    """Limpa ruído sem apagar a forma como o OCR leu o conteúdo."""
    texto = str(valor or "").replace("\x00", " ")
    texto = re.sub(r"[\t\r ]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip(" \n\t|;,")


def normalizar_busca(valor: str) -> str:
    """Cria uma chave estável, sem acentos e sem diferença de maiúsculas."""
    texto = unidecode(tratar_valor(valor)).upper()
    texto = re.sub(r"[^A-Z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()
