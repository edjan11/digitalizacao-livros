from __future__ import annotations

import re
import time
import logging

import cv2
import numpy as np

from .base import OCRProvider, OCRResult, OCRToken
from .tesseract_engine import TesseractEngine
from .rapidocr_engine import RapidOCREngine

logger = logging.getLogger(__name__)

LARGURA_OCR = 1500
LARGURA_RAPIDA = 900


def _redimensionar(image: np.ndarray, largura: int) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if w <= largura:
        return image
    escala = largura / w
    return cv2.resize(image, (largura, max(1, int(h * escala))), interpolation=cv2.INTER_AREA)


def _tem_rotulo_termo(texto: str) -> bool:
    """Indica se a leitura ja trouxe um numero precedido de rotulo de termo."""
    return re.search(
        r"(?i)(?:termo|numero|n[uú]mero|n[º°])\s*[:.]?\s*\d", texto or ""
    ) is not None


def _binarizar_para_digitos(image: np.ndarray) -> np.ndarray:
    """Versao ampliada e binarizada (Otsu) para um segundo passe de socorro.

    Fotos reais trazem o numero do termo em traco claro sobre papel curvado.
    Ampliar 1,5x devolve resolucao aos algarismos e o Otsu separa tinta de
    fundo sombreado; o modo esparso (psm 11) le numeros fora de bloco de
    paragrafo, que o psm 6 costuma engolir.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)


def _extrair_tokens_avancado(texto: str) -> list[OCRToken]:
    import re
    tokens = []

    for m in re.finditer(r'(?i)(?:termo|n[º°])\s*[:.]?\s*(\d{1,7})', texto):
        tokens.append(OCRToken(tipo="termo", valor=m.group(1), confianca=0.9))

    for m in re.finditer(r'\b(\d{6})\b', texto):
        num = m.group(1)
        if any(kw in texto.lower() for kw in ['ficha', 'cartorio', 'cartório', 'oficio', 'ofício']):
            tokens.append(OCRToken(tipo="ficha", valor=num, confianca=0.85))
        elif not any(t.valor == num for t in tokens):
            tokens.append(OCRToken(tipo="numero_grande", valor=num, confianca=0.6))

    for m in re.finditer(r'(?i)(?:folha|fls?\.?|f[º°])\s*[:.]?\s*(\d{1,4})', texto):
        tokens.append(OCRToken(tipo="folha", valor=m.group(1), confianca=0.8))

    for m in re.finditer(r'\b(19\d{2}|20\d{2})\b', texto):
        tokens.append(OCRToken(tipo="ano", valor=m.group(1), confianca=0.9))

    for m in re.finditer(r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})', texto):
        tokens.append(OCRToken(tipo="data", valor=f"{m.group(1)}/{m.group(2)}/{m.group(3)}", confianca=0.85))

    for m in re.finditer(r'(?i)(?:tabeli[aã]o|tabeliao)\s*[:.]?\s*(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})?', texto):
        if m.group(1):
            tokens.append(OCRToken(tipo="data_tabeliao", valor=m.group(1), confianca=0.9))

    for m in re.finditer(r'(?i)(?:nome|sexo|cpg|rg|data|nasc|filiac)\s*[:.]?\s*([A-ZÀ-Ú\s]{2,})', texto):
        tokens.append(OCRToken(tipo="campo_titulo", valor=m.group(0).strip(), confianca=0.7))

    for m in re.finditer(r'\b(\d{4,5})\b', texto):
        num = m.group(1)
        if not any(t.valor == num for t in tokens):
            tokens.append(OCRToken(tipo="numero", valor=num, confianca=0.4))

    return tokens


class TesseractProvider(OCRProvider):
    def __init__(self, tesseract_path: str | None = None, lang: str = "por") -> None:
        self._engine = TesseractEngine(tesseract_path=tesseract_path or None, lang=lang)

    @property
    def name(self) -> str:
        return "tesseract"

    def is_available(self) -> bool:
        return self._engine.is_available()

    def recognize(self, image, fast: bool = False) -> OCRResult:
        t0 = time.perf_counter()
        result = OCRResult(motor=self.name)
        try:
            img = _redimensionar(image, LARGURA_RAPIDA if fast else LARGURA_OCR)
            if isinstance(img, np.ndarray):
                texto = self._engine.read_array(img, psm=6)
            else:
                texto = self._engine.read(img, psm=6)
            # Segundo passe de socorro: sem rotulo de termo no texto, tenta
            # novamente sobre versao binarizada e ampliada em modo esparso.
            # O custo extra so existe quando a leitura normal nao achou o
            # numero, que e exatamente o caso em que ele vale a pena.
            if (
                isinstance(img, np.ndarray)
                and not fast
                and not _tem_rotulo_termo(texto)
            ):
                try:
                    reforco = self._engine.read_array(
                        _binarizar_para_digitos(img), psm=11
                    )
                    if reforco.strip():
                        texto = f"{texto}\n{reforco}"
                        logger.info("Tesseract: segundo passe encontrou texto adicional")
                except Exception as exc:
                    logger.warning("Segundo passe Tesseract falhou: %s", exc)
            result.texto_bruto = texto.strip()
            tokens = _extrair_tokens_avancado(texto.strip())
            for t in tokens:
                t.motor = self.name
            result.tokens = tokens
        except Exception as e:
            logger.warning("Tesseract OCR falhou: %s", e)
        result.tempo_ms = (time.perf_counter() - t0) * 1000
        return result


class RapidOCRProvider(OCRProvider):
    def __init__(self, cache_engine: bool = True, apenas_cabecalhos: bool = False) -> None:
        self._engine = None
        self._cache = cache_engine
        self._apenas_cabecalhos = apenas_cabecalhos

    @property
    def name(self) -> str:
        return "rapidocr"

    def is_available(self) -> bool:
        try:
            if self._engine is None:
                self._engine = RapidOCREngine()
                self._engine.load()
            return True
        except Exception:
            return False

    def recognize(self, image, fast: bool = False) -> OCRResult:
        t0 = time.perf_counter()
        result = OCRResult(motor=self.name)
        try:
            if self._engine is None:
                self._engine = RapidOCREngine()
                self._engine.load()
            if self._apenas_cabecalhos and isinstance(image, np.ndarray):
                img = self._recortar_cabecalhos(image)
            else:
                img = _redimensionar(image, LARGURA_RAPIDA if fast else LARGURA_OCR)
            boxes = self._engine.read_with_boxes(img)
            texto_parts = []
            texto_tokens = []
            altura, largura = img.shape[:2]
            for bbox, text, conf in boxes:
                texto_parts.append(text)
                bbox_normalizada = [
                    [round(float(p[0]) / max(1, largura), 6), round(float(p[1]) / max(1, altura), 6)]
                    for p in bbox
                ] if bbox else None
                texto_tokens.append(
                    OCRToken(
                        tipo="texto_linha",
                        valor=str(text),
                        confianca=float(conf),
                        motor=self.name,
                        bbox=bbox_normalizada,
                    )
                )
            result.texto_bruto = "\n".join(texto_parts)
            tokens = _extrair_tokens_avancado(result.texto_bruto)
            for t in tokens:
                t.motor = self.name
            result.tokens = texto_tokens + tokens
        except Exception as e:
            logger.warning("RapidOCR falhou: %s", e)
        result.tempo_ms = (time.perf_counter() - t0) * 1000
        return result

    @staticmethod
    def _recortar_cabecalhos(image: np.ndarray) -> np.ndarray:
        """Mantem as duas faixas que contem ``Numero`` nos livros A-07.

        O formulario tem dois registros por face. Processar essas faixas em vez
        dos 24 MP da fotografia preserva os algarismos e reduz bastante o custo
        do detector do RapidOCR. A faixa larga tolera pequenas mudancas de
        enquadramento e da divisoria entre os registros.
        """
        h, w = image.shape[:2]
        # A coluna "Número" varia bastante com a curvatura e o enquadramento.
        # A faixa antiga (27% x 12%) perdia o cabeçalho em fotos reais do A-07.
        x_final = max(1, int(w * 0.36))
        topo = image[:max(1, int(h * 0.18)), :x_final]
        inferior = image[int(h * 0.43):max(int(h * 0.43) + 1, int(h * 0.62)), :x_final]
        recorte = cv2.vconcat([topo, inferior])
        largura = 550
        if recorte.shape[1] > largura:
            escala = largura / recorte.shape[1]
            recorte = cv2.resize(
                recorte,
                (largura, max(1, int(recorte.shape[0] * escala))),
                interpolation=cv2.INTER_AREA,
            )
        return recorte
