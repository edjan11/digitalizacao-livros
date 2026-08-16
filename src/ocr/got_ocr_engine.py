from __future__ import annotations

import importlib
import logging
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config.settings import app_root, data_dir, is_frozen
from .base import OCRProvider, OCRResult, OCRToken

logger = logging.getLogger(__name__)

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
MODEL_DIR_NAME = "got-ocr-2.0-hf"
MODEL_SIZE_MIB = 1088

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def caminho_modelo_got(model_path: str | Path | None = None) -> Path:
    if model_path:
        return Path(model_path)
    if not is_frozen():
        local = app_root() / "models" / MODEL_DIR_NAME
        if local.exists():
            return local
    return data_dir() / "models" / MODEL_DIR_NAME


def modelo_got_instalado(model_path: str | Path | None = None) -> bool:
    pasta = caminho_modelo_got(model_path)
    return (pasta / "config.json").is_file() and (pasta / "model.safetensors").is_file()


def _imagem_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    carregada = Image.open(str(image))
    return carregada.convert("RGB")


def _tokens_do_texto(texto: str, motor: str = "got-ocr2") -> list[OCRToken]:
    """Transforma a transcricao generativa em metadados sem inventar caixas.

    O GOT devolve texto corrido, não bounding boxes nem uma confiança calibrada.
    Por isso as linhas entram com confiança conservadora e ficam disponíveis
    para confirmação/correção humana na Consulta.
    """
    tokens: list[OCRToken] = []
    for linha in texto.splitlines():
        valor = re.sub(r"\s+", " ", linha).strip()
        if valor:
            tokens.append(
                OCRToken(
                    tipo="texto_linha",
                    valor=valor,
                    confianca=0.55,
                    motor=motor,
                )
            )

    for match in re.finditer(
        r"(?i)(?:termo|n[uú]mero|n[º°])\s*[:.]?\s*(\d{1,7})", texto
    ):
        tokens.append(
            OCRToken("termo", match.group(1), 0.68, motor)
        )
    for match in re.finditer(
        r"(?i)(?:folha|fls?\.?|f[º°])\s*[:.]?\s*(\d{1,4})", texto
    ):
        tokens.append(
            OCRToken("folha", match.group(1), 0.64, motor)
        )
    for match in re.finditer(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)", texto):
        tokens.append(
            OCRToken("ano", match.group(1), 0.58, motor)
        )
    return tokens


def _texto_repetitivo(texto: str) -> bool:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    if len(linhas) < 8:
        return False
    mais_repetida = max(linhas.count(linha) for linha in set(linhas))
    return mais_repetida / len(linhas) >= 0.70


class GOTOCRProvider(OCRProvider):
    """GOT-OCR 2.0 oficial, carregado sob demanda e executado localmente.

    Nesta estação a execução é CPU-only. O provider não participa da captura;
    ele é usado somente pela indexação secundária da Consulta do Acervo.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        permitir_download: bool = False,
        max_new_tokens: int = 384,
    ) -> None:
        self.model_path = caminho_modelo_got(model_path)
        self.permitir_download = permitir_download
        self.max_new_tokens = max(64, min(int(max_new_tokens), 4096))
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "got-ocr2"

    def is_available(self) -> bool:
        try:
            importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            return hasattr(transformers, "GotOcr2ForConditionalGeneration")
        except (ImportError, RuntimeError):
            return False

    def _baixar_modelo(self) -> None:
        if modelo_got_instalado(self.model_path):
            return
        if not self.permitir_download:
            raise RuntimeError(
                "O modelo GOT-OCR 2.0 ainda não foi baixado. "
                "Ative a opção na Consulta para autorizar o download."
            )
        from huggingface_hub import snapshot_download

        self.model_path.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando %s para %s", MODEL_ID, self.model_path)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(self.model_path),
        )

    def load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "GOT-OCR 2.0 indisponível: verifique transformers, tokenizers e torch."
            )
        self._baixar_modelo()

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Carregando GOT-OCR 2.0 em CPU a partir de %s", self.model_path)
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        self._model.to("cpu")
        self._model.eval()

    def recognize(self, image, fast: bool = False) -> OCRResult:
        inicio = time.perf_counter()
        self.load()

        import torch

        imagem = _imagem_pil(image)
        inputs = self._processor(imagem, return_tensors="pt")
        with torch.inference_mode():
            gerados = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=self.max_new_tokens,
            )
        inicio_resposta = inputs["input_ids"].shape[1]
        texto = self._processor.decode(
            gerados[0, inicio_resposta:], skip_special_tokens=True
        ).strip()
        if _texto_repetitivo(texto):
            raise RuntimeError("GOT-OCR gerou uma resposta repetitiva e foi descartado")
        return OCRResult(
            tokens=_tokens_do_texto(texto, self.name),
            texto_bruto=texto,
            motor=self.name,
            tempo_ms=(time.perf_counter() - inicio) * 1000,
        )
