from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .hashing import hash_distance

logger = logging.getLogger(__name__)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1 is None or img2 is None:
        return 0.0
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    h = min(gray1.shape[0], gray2.shape[0])
    w = min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (w, h))
    gray2 = cv2.resize(gray2, (w, h))
    try:
        return float(ssim(gray1, gray2, data_range=255))
    except Exception:
        return 0.0


def compute_registered_similarity(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Compara duas capturas depois de alinhar o formulario.

    O SSIM puro tende a considerar todas as folhas do mesmo livro parecidas,
    porque as linhas impressas ocupam boa parte da imagem. Alem do SSIM, esta
    rotina mede a coincidencia da tinta depois de remover linhas horizontais e
    verticais longas do formulario.
    """
    if img1 is None or img2 is None:
        return {"ssim": 0.0, "alinhamento": 0.0, "tinta": 0.0}

    tamanho = (256, 384)
    a = cv2.resize(img1, tamanho, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img2, tamanho, interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    ga_float = ga.astype(np.float32) / 255.0
    gb_float = gb.astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    alinhamento = 0.0
    try:
        alinhamento, warp = cv2.findTransformECC(
            ga_float,
            gb_float,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
            None,
            3,
        )
    except cv2.error:
        logger.debug("Nao foi possivel alinhar imagens para duplicidade", exc_info=True)

    b_alinhada = cv2.warpAffine(
        b,
        warp,
        tamanho,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    gb_alinhada = (
        cv2.cvtColor(b_alinhada, cv2.COLOR_BGR2GRAY)
        if b_alinhada.ndim == 3
        else b_alinhada
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ga_norm = clahe.apply(ga)
    gb_norm = clahe.apply(gb_alinhada)
    try:
        ssim_val = float(ssim(ga_norm, gb_norm, data_range=255))
    except Exception:
        ssim_val = 0.0

    def mascara_tinta(gray: np.ndarray) -> np.ndarray:
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )
        horizontais = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        )
        verticais = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        )
        mask[(horizontais > 0) | (verticais > 0)] = 0
        return mask

    tinta_a = mascara_tinta(ga_norm)
    tinta_b = mascara_tinta(gb_norm)
    distancia_a = cv2.distanceTransform(255 - tinta_a, cv2.DIST_L2, 3)
    distancia_b = cv2.distanceTransform(255 - tinta_b, cv2.DIST_L2, 3)
    pontos_a = tinta_a > 0
    pontos_b = tinta_b > 0
    if not np.any(pontos_a) or not np.any(pontos_b):
        tinta = 0.0
    else:
        a_em_b = float((distancia_b[pontos_a] <= 1.5).mean())
        b_em_a = float((distancia_a[pontos_b] <= 1.5).mean())
        tinta = (a_em_b + b_em_a) / 2.0

    return {
        "ssim": max(0.0, min(1.0, ssim_val)),
        "alinhamento": max(0.0, min(1.0, float(alinhamento))),
        "tinta": max(0.0, min(1.0, tinta)),
    }


def comparar_phash(h1: str, h2: str, threshold: int = 5) -> bool:
    return hash_distance(h1, h2) <= threshold


def comparar_dhash(h1: str, h2: str, threshold: int = 8) -> bool:
    return hash_distance(h1, h2) <= threshold
