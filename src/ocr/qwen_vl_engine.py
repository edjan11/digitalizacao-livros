from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config.settings import app_root, data_dir, is_frozen
from .base import OCRResult, OCRToken

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_DIR_NAME = "qwen2-vl-2b-instruct"
MODEL_SIZE_MIB = 4225
PIXELS_MIN_PADRAO = 128 * 28 * 28
PIXELS_MAX_PADRAO = 384 * 28 * 28

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def caminho_modelo_qwen(model_path: str | Path | None = None) -> Path:
    if model_path:
        return Path(model_path)
    if not is_frozen():
        local = app_root() / "models" / MODEL_DIR_NAME
        if local.exists():
            return local
    return data_dir() / "models" / MODEL_DIR_NAME


def modelo_qwen_instalado(model_path: str | Path | None = None) -> bool:
    pasta = caminho_modelo_qwen(model_path)
    pesos = list(pasta.glob("model*.safetensors"))
    return (pasta / "config.json").is_file() and bool(pesos)


def _imagem_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return Image.open(str(image)).convert("RGB")


def _limpar_resposta(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"^```(?:text|plaintext)?\s*", "", texto, flags=re.I)
    texto = re.sub(r"\s*```$", "", texto)
    if len(texto) >= 2 and texto[0] == texto[-1] and texto[0] in "\"'":
        texto = texto[1:-1].strip()
    return texto


def _campos_json_qwen(texto: str) -> dict[str, str]:
    """Extrai os três campos mesmo quando o modelo envolve o JSON em markdown."""
    limpo = _limpar_resposta(texto)
    inicio, fim = limpo.find("{"), limpo.rfind("}")
    candidato = limpo[inicio : fim + 1] if inicio >= 0 and fim > inicio else limpo
    try:
        dados = json.loads(candidato)
    except (TypeError, ValueError, json.JSONDecodeError):
        dados = {}
    if not isinstance(dados, dict):
        dados = {}
    aliases = {
        "nome_registrado": ("nome_registrado", "nome", "registrado"),
        "termo": ("termo", "numero_termo", "número_termo", "numero"),
        "nome_mae": ("nome_mae", "nome_mãe", "mae", "mãe"),
        "data_registro": (
            "data_registro", "data_do_registro", "data", "data_ato",
        ),
    }
    saida: dict[str, str] = {}
    for campo, chaves in aliases.items():
        valor = next((dados.get(chave) for chave in chaves if dados.get(chave)), "")
        saida[campo] = str(valor).strip() if valor is not None else ""
    # O modelo pode atingir o limite logo depois de escrever a data e deixar o
    # JSON sem a aspa ou a chave final. Preserve os pares que ele jÃ¡ produziu.
    if not all(saida.values()):
        for campo, chaves in aliases.items():
            if saida[campo]:
                continue
            for chave in chaves:
                trecho = re.search(
                    rf'"{re.escape(chave)}"\s*:\s*"((?:\\.|[^"\\])*)',
                    limpo,
                    flags=re.I,
                )
                if not trecho:
                    continue
                bruto = trecho.group(1)
                try:
                    valor = json.loads('"' + bruto + '"')
                except (TypeError, ValueError, json.JSONDecodeError):
                    valor = bruto.replace('\\"', '"').replace('\\\\', '\\')
                saida[campo] = str(valor).strip()
                break
    return saida


def preparar_imagem_qwen(image: np.ndarray) -> np.ndarray:
    """Remove sombra com contraste local, sem binarizar nem alterar o original.

    O recorte já é feito pelo operador em coordenadas da foto nativa. Esta
    função só produz uma cópia temporária para a inferência: preserva a tinta
    azul e as linhas do formulário melhor que um threshold preto/branco.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class QwenAreaAnalyzer:
    """Leitura contextual de uma pequena área escolhida pelo operador.

    O Qwen2-VL 2B não é executado em lote nesta estação. A área explícita evita
    que datas de averbações e outros campos sejam atribuídos ao registro errado.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        permitir_download: bool = False,
        max_new_tokens: int = 96,
        min_pixels: int = PIXELS_MIN_PADRAO,
        max_pixels: int = PIXELS_MAX_PADRAO,
        dtype: str = "auto",
        threads: int = 0,
    ) -> None:
        self.model_path = caminho_modelo_qwen(model_path)
        self.permitir_download = permitir_download
        self.max_new_tokens = max(16, min(int(max_new_tokens), 256))
        self.min_pixels = max(28 * 28, int(min_pixels))
        self.max_pixels = max(self.min_pixels, int(max_pixels))
        self.dtype_name = str(dtype or "auto").lower()
        self.threads = max(0, int(threads))
        self._processor = None
        self._model = None

    @staticmethod
    def _dtype_cpu(torch):
        """Escolhe dtype compatível com a ISA real da CPU.

        Xeon/desktop AVX2 não possui instruções nativas BF16; nesse caso
        float32 costuma ser mais rápido e há memória suficiente no computador
        de captura. BF16 fica reservado para CPUs que anunciam AVX512.
        """
        if torch.backends.cpu.get_cpu_capability() == "AVX512":
            return torch.bfloat16
        return torch.float32

    def is_available(self) -> bool:
        try:
            importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            return hasattr(transformers, "Qwen2VLForConditionalGeneration")
        except (ImportError, RuntimeError):
            return False

    def _baixar_modelo(self) -> None:
        if modelo_qwen_instalado(self.model_path):
            return
        if not self.permitir_download:
            raise RuntimeError(
                "O modelo Qwen2-VL 2B ainda não foi baixado. "
                "Autorize o download pela Consulta."
            )
        from huggingface_hub import snapshot_download

        self.model_path.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando %s para %s", MODEL_ID, self.model_path)
        snapshot_download(repo_id=MODEL_ID, local_dir=str(self.model_path))

    def load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError("Qwen2-VL indisponível neste ambiente.")
        self._baixar_modelo()

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        threads = self.threads or min(24, int(os.cpu_count() or 1))
        torch.set_num_threads(max(1, threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, threads)))
        except RuntimeError:
            # O processo pode já ter executado outra operação Torch; nesse
            # caso o número de interop fica com o valor que ele já escolheu.
            pass

        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        dtype = self._dtype_cpu(torch) if self.dtype_name == "auto" else getattr(
            torch, self.dtype_name
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=False,
        )
        self._model.to("cpu")
        self._model.eval()

    def analisar(self, image, instrucao: str, tipo: str) -> OCRResult:
        inicio = time.perf_counter()
        self.load()

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instrucao},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt],
            images=[_imagem_pil(image)],
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        respostas = [
            saida[len(entrada):]
            for entrada, saida in zip(inputs.input_ids, ids)
        ]
        texto = self._processor.batch_decode(
            respostas,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        texto = _limpar_resposta(texto)
        return OCRResult(
            tokens=[
                OCRToken(
                    tipo=tipo,
                    valor=texto,
                    confianca=0.45,
                    motor="qwen2-vl-2b-area",
                )
            ] if texto else [],
            texto_bruto=texto,
            motor="qwen2-vl-2b-area",
            tempo_ms=(time.perf_counter() - inicio) * 1000,
        )

    def liberar(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()


class QwenRecordAnalyzer(QwenAreaAnalyzer):
    """Uma inferência por registro para os campos básicos auditáveis."""

    def analisar_nome(self, image) -> tuple[str, OCRResult]:
        """Lê somente a faixa estreita do nome, sem processar o assento todo."""
        resultado = self.analisar(
            image,
            instrucao=(
                "Esta imagem contém somente a linha de um registro civil depois de "
                "'que recebeu o nome de'. Transcreva apenas o nome completo manuscrito "
                "da pessoa registrada. Não inclua rótulos, sexo, pais, datas, termo, "
                "pontuação final nem explicações. Não invente letras ilegíveis."
            ),
            tipo="nome_registrado",
        )
        return _limpar_resposta(resultado.texto_bruto), resultado

    def analisar_registro(self, image) -> tuple[dict[str, str], OCRResult]:
        instrucao = (
            "Leia somente este registro civil manuscrito em português. "
            "Extraia apenas quatro campos: o nome completo do registrado, o número "
            "do termo, o nome completo da mãe e a data do registro. A data do registro "
            "é a data escrita no cabeçalho depois de 'Em' e antes de 'de mil novecentos'; "
            "não use a data do nascimento, averbação ou assinatura. O nome da mãe normalmente aparece "
            "logo depois do nome do registrado, no trecho que começa por 'filho' "
            "ou 'filha'. Ignore a coluna 'Notas, Averbações e Retificações', "
            "assinaturas e o segundo registro da página. Não invente letras ilegíveis. "
            "Responda exclusivamente neste JSON válido, sem markdown: "
            '{"nome_registrado":"", "termo":"", "nome_mae":"", "data_registro":""}'
        )
        resultado = self.analisar(
            image,
            instrucao=instrucao,
            tipo="registro_basico",
        )
        return _campos_json_qwen(resultado.texto_bruto), resultado
