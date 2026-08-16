"""Conversao PDF -> JPG em alta qualidade (300 DPI, RGB, Q95, 4:4:4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import cv2

SUPPORTED_PDF_EXTS = {".pdf"}


class ConversionError(Exception):
    pass


@dataclass
class PageImage:
    index: int  # pagina (0-based)
    path: Path
    width: int
    height: int
    bytes_size: int


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if False else image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def pdf_to_jpgs(
    source: Path,
    output_dir: Path,
    dpi: int = 300,
    jpeg_quality: int = 95,
    keep_jpeg_original: bool = True,
) -> list[PageImage]:
    """Converte todas as paginas de um PDF em JPGs de alta qualidade.

    Se a pagina ja e uma imagem JPEG embutida, extrai sem recompressao.
    """
    if source.suffix.lower() not in SUPPORTED_PDF_EXTS:
        raise ConversionError(f"Formato nao suportado: {source.suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PageImage] = []

    try:
        doc = fitz.open(source)
    except Exception as exc:
        raise ConversionError(f"Nao foi possivel abrir PDF {source.name}: {exc}") from exc

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            dest = output_dir / f"page_{page_index + 1:04d}.jpg"

            # Tenta extrair imagem JPEG original da pagina (evita recompressao)
            wrote = False
            if keep_jpeg_original:
                try:
                    images = page.get_images(full=True)
                    if images:
                        for img_info in images:
                            xref = img_info[0]
                            pix = fitz.Pixmap(doc, xref)
                            if pix.n >= 3 and pix.alpha == 0 and pix.colorspace and "jpeg" in getattr(pix.colorspace, "name", "") or pix.n - (1 if pix.alpha else 0) == 3:
                                base = doc.extract_image(xref)
                                if base and base.get("ext") == "jpeg":
                                    dest.write_bytes(base["image"])
                                    wrote = True
                                    break
                            pix = None
                except Exception:
                    wrote = False

            if not wrote:
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                np_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                rgb = _to_rgb(np_img) if pix.n >= 3 else cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(
                    ".jpg",
                    rgb,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if not ok:
                    raise ConversionError(f"Falha ao codificar JPG da pagina {page_index + 1}")
                dest.write_bytes(buf.tobytes())

            img = cv2.imread(str(dest))
            if img is None:
                raise ConversionError(f"JPG gerado invalido: {dest.name}")
            height, width = img.shape[:2]
            results.append(
                PageImage(
                    index=page_index,
                    path=dest,
                    width=width,
                    height=height,
                    bytes_size=dest.stat().st_size,
                )
            )
    except Exception as exc:
        raise ConversionError(str(exc)) from exc
    finally:
        doc.close()

    return results