"""Normalização e extração de metadados pesquisáveis do OCR."""

from .extractor import MetadataDetection, extrair_metadados
from .normalizer import normalizar_busca, tratar_valor

__all__ = [
    "MetadataDetection",
    "extrair_metadados",
    "normalizar_busca",
    "tratar_valor",
]
