from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import imagehash
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def compute_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def _pil_cinza(image: cv2.Mat) -> Image.Image | None:
    if image is None or image.size == 0:
        return None
    cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return Image.fromarray(cinza)


def compute_phash(image: cv2.Mat) -> str:
    pil_img = _pil_cinza(image)
    if pil_img is None:
        return ""
    return str(imagehash.phash(pil_img))


def compute_dhash(image: cv2.Mat) -> str:
    pil_img = _pil_cinza(image)
    if pil_img is None:
        return ""
    return str(imagehash.dhash(pil_img))


def compute_hashes(image: cv2.Mat) -> tuple[str, str]:
    """Calcula os dois hashes compartilhando a conversao da foto para cinza."""
    pil_img = _pil_cinza(image)
    if pil_img is None:
        return "", ""
    return str(imagehash.phash(pil_img)), str(imagehash.dhash(pil_img))


def hash_distance(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 999
    if h1 == h2:
        return 0
    h1_int = int(h1, 16) if not h1.startswith("0x") else int(h1, 16)
    h2_int = int(h2, 16) if not h2.startswith("0x") else int(h2, 16)
    return bin(h1_int ^ h2_int).count("1")
