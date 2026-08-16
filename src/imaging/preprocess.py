from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_ocr(
    image: np.ndarray,
    source_path: str | None = None,
) -> np.ndarray:
    """Normaliza somente o formato, sem realinhar ou recriar a fotografia.

    A orientação validada fica em ``imagem.rotacao_visualizacao`` e é aplicada
    pelo chamador. Deskew/OSD recorrente foi removido: além de lento, podia
    modificar a geometria a cada leitura. Como o OCR é persistente e executado
    uma vez, também não são criados arquivos ``.ocr_cache`` junto aos originais.
    ``source_path`` permanece no contrato para compatibilidade com chamadas
    existentes.
    """
    del source_path
    if image is None or image.size == 0:
        return image
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
