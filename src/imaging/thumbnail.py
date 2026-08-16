from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def gerar_thumbnail(image: np.ndarray, width: int = 200) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((width, width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = width / w
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def gerar_thumbnail_arquivo(source_path: Path, dest_path: Path, width: int = 200) -> Path:
    img = cv2.imread(str(source_path))
    if img is None:
        raise ValueError(f"Nao foi possivel ler: {source_path}")
    thumb = gerar_thumbnail(img, width)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dest_path
