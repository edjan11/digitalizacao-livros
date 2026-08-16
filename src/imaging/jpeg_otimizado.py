"""Codificacao JPEG otimizada: melhor qualidade visual com menor tamanho.

Regra de ouro deste projeto: o OCR SEMPRE processa a imagem full-res sem
perda. Esta otimizacao aplica-se APENAS ao JPG final gravado no sistema
(acervo), nunca a imagem usada durante o processamento.

Tecnicas aplicadas (todas sem perda perceptivel de qualidade):
- IMWRITE_JPEG_OPTIMIZE: otimizacao Huffman do libjpeg (gratis, ~5-10% menor)
- IMWRITE_JPEG_SAMPLING_FACTOR_420: chroma subsampling 4:2:0 (padrao do libjpeg;
  irrelevante para texto/linhas, que vivem no canal luma)
- Qualidade configuravel (default 90: ponto otimo qualidade/tamanho)
"""

from __future__ import annotations

import cv2
import numpy as np

# Flags idempotentes: OpenCV 5 usa constantes novas; OpenCV antigo usa numeros.
_QUALITY = int(cv2.IMWRITE_JPEG_QUALITY)
_OPTIMIZE = int(cv2.IMWRITE_JPEG_OPTIMIZE)
_SAMPLING = int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR)
_SAMPLING_420 = int(getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_420", 2))

QUALIDADE_PADRAO = 90


def _params(quality: int) -> list[int]:
    """Monta os parametros de codificacao compatíveis com a versao do OpenCV.

    OpenCV 5 recusa o valor inteiro 2 para IMWRITE_JPEG_SAMPLING_FACTOR e exige
    a constante IMWRITE_JPEG_SAMPLING_FACTOR_420; versoes antigas aceitam 2.
    Usamos o que a instalacao expuser.
    """
    quality = max(1, min(100, int(quality)))
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_420"):
        return [
            _QUALITY, quality,
            _OPTIMIZE, 1,
            _SAMPLING, _SAMPLING_420,
        ]
    return [
        _QUALITY, quality,
        _OPTIMIZE, 1,
        _SAMPLING, 2,
    ]


def codificar_jpeg(image: np.ndarray, quality: int = QUALIDADE_PADRAO) -> bytes:
    """Codifica a imagem como JPG otimizado e retorna os bytes."""
    ok, buf = cv2.imencode(".jpg", image, _params(quality))
    if not ok:
        raise ValueError("Falha ao codificar imagem como JPEG")
    return buf.tobytes()


def gravar_jpeg(dest, image: np.ndarray, quality: int = QUALIDADE_PADRAO) -> int:
    """Escreve o JPG otimizado em `dest` (path ou str). Retorna bytes gravados."""
    data = codificar_jpeg(image, quality)
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)
