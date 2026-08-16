"""Reproduz a leitura do nome manuscrito do termo 6801 sem alterar o banco.

Uso (CPU, aproximadamente 3 minutos):
    .venv\\Scripts\\python.exe scripts\\prova_qwen_6801.py

O retângulo é o salvo no revisor em coordenadas relativas da foto original.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metadata.normalizer import normalizar_busca
from src.ocr.qwen_vl_engine import QwenAreaAnalyzer, preparar_imagem_qwen


DEFAULT_IMAGE = Path(r"D:\A - 07\FRENTE\IMG_2025_07_02_14_14_07S.jpg")
BBOX_NOME = (0.27, 0.155, 0.80, 0.202)
ESPERADO = "Anderson da Silva Cruz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--contraste",
        action="store_true",
        help="compara a cópia CLAHE experimental; não é o modo padrão",
    )
    args = parser.parse_args()
    dados = np.fromfile(str(args.imagem), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise SystemExit(f"Imagem não encontrada ou inválida: {args.imagem}")
    altura, largura = imagem.shape[:2]
    x1, y1 = int(BBOX_NOME[0] * largura), int(BBOX_NOME[1] * altura)
    x2, y2 = int(BBOX_NOME[2] * largura), int(BBOX_NOME[3] * altura)
    recorte = imagem[y1:y2, x1:x2]
    enviado = preparar_imagem_qwen(recorte) if args.contraste else recorte
    inicio = time.perf_counter()
    motor = QwenAreaAnalyzer(permitir_download=False, max_new_tokens=96)
    resultado = motor.analisar(
        enviado,
        instrucao=(
            "Transcreva exatamente o nome manuscrito completo nesta área. "
            "Responda somente com o nome, sem explicação."
        ),
        tipo="nome_registrado",
    )
    motor.liberar()
    texto = resultado.texto_bruto.strip()
    ok = normalizar_busca(texto) == normalizar_busca(ESPERADO)
    print(f"imagem: {args.imagem}")
    print(f"recorte original: x={x1}:{x2}, y={y1}:{y2} ({recorte.shape[1]}x{recorte.shape[0]})")
    print(f"variante: {'contraste CLAHE' if args.contraste else 'original (padrão)'}")
    print(f"motor: {resultado.motor} | tempo: {resultado.tempo_ms / 1000:.1f}s")
    print(f"texto extraído: {texto}")
    print(f"esperado conferido visualmente: {ESPERADO}")
    print("RESULTADO: " + ("PASSOU" if ok else "DIVERGIU — revisar no revisor"))


if __name__ == "__main__":
    main()
