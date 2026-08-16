from __future__ import annotations

import time
import logging
import re
import os

import cv2
import numpy as np

from .base import OCRProvider, OCRResult, OCRToken

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

LARGURA_HTR = 1200


def _redimensionar(image: np.ndarray, largura: int = LARGURA_HTR) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if w <= largura:
        return image
    escala = largura / w
    return cv2.resize(image, (largura, max(1, int(h * escala))), interpolation=cv2.INTER_AREA)


def _extrair_tokens(texto: str) -> list[OCRToken]:
    tokens = []
    for m in re.finditer(r'\b(\d{3,7})\b', texto):
        val = m.group(1)
        tokens.append(OCRToken(tipo="termo", valor=val, confianca=0.55))
    for m in re.finditer(r'(?i)(?:folha|fls?\.?|f[º°])\s*[:.]?\s*(\d{1,4})', texto):
        tokens.append(OCRToken(tipo="folha", valor=m.group(1), confianca=0.7))
    for m in re.finditer(r'\b(19\d{2}|20\d{2})\b', texto):
        tokens.append(OCRToken(tipo="ano", valor=m.group(1), confianca=0.7))
    for m in re.finditer(r'(?i)(?:n[º°])\s*(\d{1,7})', texto):
        tokens.append(OCRToken(tipo="termo", valor=m.group(1), confianca=0.65))
    for m in re.finditer(r'(?i)(?:livro|lv\.?)\s*[:.]?\s*([A-Za-z0-9\-]+)', texto):
        tokens.append(OCRToken(tipo="livro", valor=m.group(1), confianca=0.6))
    return tokens


class EasyOCRProvider(OCRProvider):
    def __init__(self, idiomas: list[str] | None = None) -> None:
        self._reader = None
        self._idiomas = idiomas or ["pt", "en"]

    @property
    def name(self) -> str:
        return "easyocr"

    def is_available(self) -> bool:
        try:
            import importlib
            importlib.import_module("easyocr")
            return True
        except ImportError:
            return False

    def load(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr
            logger.info("Carregando EasyOCR (idiomas: %s)...", self._idiomas)
            self._reader = easyocr.Reader(
                self._idiomas,
                gpu=False,
                verbose=False,
            )
            logger.info("EasyOCR carregado com sucesso")
        except Exception as e:
            logger.warning("Falha ao carregar EasyOCR: %s", str(e)[:120])

    def recognize(self, image, fast: bool = False) -> OCRResult:
        t0 = time.perf_counter()
        result = OCRResult(motor=self.name)
        try:
            self.load()
            if self._reader is None:
                return result

            img = _redimensionar(image)
            if isinstance(img, np.ndarray):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_bgr = cv2.imread(str(image))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None

            if img_rgb is None:
                return result

            raw_results = self._reader.readtext(img_rgb, detail=1)
            lines = []
            for bbox, text, conf in raw_results:
                lines.append(text)
                result.tokens.append(OCRToken(
                    tipo="texto",
                    valor=str(text).strip(),
                    confianca=float(conf),
                    motor=self.name,
                    bbox=list(bbox) if bbox else None,
                ))

            result.texto_bruto = "\n".join(lines)
            tokens = _extrair_tokens("\n".join(lines))
            for t in tokens:
                t.motor = self.name
                t.confianca = min(t.confianca + 0.1, 1.0)
            result.tokens.extend(tokens)
        except Exception as e:
            logger.warning("EasyOCR falhou: %s", str(e)[:100])
        result.tempo_ms = (time.perf_counter() - t0) * 1000
        return result


class HTREngine(EasyOCRProvider):
    def __init__(self, idiomas: list[str] | None = None) -> None:
        super().__init__(idiomas=idiomas or ["pt", "en"])

    @property
    def name(self) -> str:
        return "htr"

    def is_available(self) -> bool:
        return super().is_available()

    def load(self) -> None:
        super().load()

    def recognize(self, image, fast: bool = False) -> OCRResult:
        return super().recognize(image, fast=fast)
