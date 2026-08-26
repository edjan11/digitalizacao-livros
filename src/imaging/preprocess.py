from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_ocr(
    image: np.ndarray,
    source_path: str | None = None,
) -> np.ndarray:
    """Normaliza formato e realca contraste, sem alterar a geometria.

    A orientação validada fica em ``imagem.rotacao_visualizacao`` e é aplicada
    pelo chamador. Deskew/OSD recorrente foi removido: além de lento, podia
    modificar a geometria a cada leitura. Como o OCR é persistente e executado
    uma vez, também não são criados arquivos ``.ocr_cache`` junto aos originais.
    ``source_path`` permanece no contrato para compatibilidade com chamadas
    existentes.

    O CLAHE leve no canal L compensa sombra de curvatura do livro sem
    binarizar: números manuscritos em página mal iluminada passam a ser lidos
    com mais frequência pelos dois motores.
    """
    del source_path
    if image is None or image.size == 0:
        return image
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    claro, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    claro = clahe.apply(claro)
    return cv2.cvtColor(cv2.merge([claro, a, b]), cv2.COLOR_LAB2BGR)
