"""Visual orientation helpers for external viewers.

The archive JPG is never rewritten.  A rotated derivative is created only
when an external viewer (for example Chrome) cannot consume the application's
``rotacao_visualizacao`` metadata itself.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


def normalizar_rotacao(rotacao: int | None) -> int:
    valor = int(rotacao or 0) % 360
    if valor not in (0, 90, 180, 270):
        return 0
    return valor


def aplicar_rotacao(image: np.ndarray, rotacao: int | None) -> np.ndarray:
    """Returns a rotated copy without modifying ``image``."""
    valor = normalizar_rotacao(rotacao)
    if image is None or image.size == 0 or valor == 0:
        return image.copy() if isinstance(image, np.ndarray) else image
    if valor == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if valor == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def materializar_copia_orientada(
    path: str | Path,
    rotacao: int | None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Creates/returns a Chrome-safe oriented derivative.

    Rotation 0 returns the original path.  Other rotations are cached using
    the source absolute path, size, mtime and rotation.  The write is atomic,
    and JPEG quality is 100; the source bytes remain untouched.
    """
    origem = Path(path).resolve()
    valor = normalizar_rotacao(rotacao)
    if valor == 0:
        return origem
    if not origem.is_file():
        raise FileNotFoundError(str(origem))
    dados = np.fromfile(str(origem), dtype=np.uint8)
    image = cv2.imdecode(dados, cv2.IMREAD_UNCHANGED) if dados.size else None
    if image is None:
        raise ValueError(f"Não foi possível decodificar {origem}")
    oriented = aplicar_rotacao(image, valor)
    stat = origem.stat()
    chave = hashlib.sha256(
        f"{origem}|{stat.st_size}|{stat.st_mtime_ns}|{valor}".encode("utf-8")
    ).hexdigest()[:24]
    destino_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "DigitalizadorLivros" / "orientadas"
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / f"{origem.stem}_{chave}_{valor}.jpg"
    if not destino.is_file():
        ok, buffer = cv2.imencode(
            ".jpg", oriented, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )
        if not ok:
            raise ValueError(f"Não foi possível codificar {origem}")
        temporario = destino.with_suffix(destino.suffix + f".{os.getpid()}.tmp")
        buffer.tofile(str(temporario))
        os.replace(temporario, destino)
    return destino
