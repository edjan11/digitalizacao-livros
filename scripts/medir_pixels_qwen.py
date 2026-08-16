"""Mede o Qwen2-VL-2B local com diferentes limites de pixels (mesmos recortes).

O objetivo e verificar a curva tempo x pixels do encoder visual: se o gargalo
e o numero de tokens de imagem, reduzir max_pixels acelera na proporcao.
Nao altera nenhum arquivo armazenado.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.qwen_vl_engine import QwenAreaAnalyzer, preparar_imagem_qwen

TERMOS = ["6801", "6802", "6805", "6810", "6838"]

INSTRUCAO = (
    "A imagem e um recorte de um registro civil brasileiro. "
    "Leia SOMENTE o nome manuscrito que vem depois das palavras "
    "impressas 'que recebeu o nome de'. Ignore todo o resto. "
    "Responda apenas com o nome completo, sem explicacao."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".tmp_qwen_20_line")
    ap.add_argument("--max-pixels", type=int, default=301056)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    analisador = QwenAreaAnalyzer(
        max_new_tokens=32,
        min_pixels=128 * 28 * 28,
        max_pixels=args.max_pixels,
        threads=args.threads,
    )
    total = 0.0
    for termo in TERMOS:
        path = next(Path(args.input).glob(f"*_{termo}.jpg"))
        ini = time.perf_counter()
        resultado = analisador.analisar(
            preparar_imagem_qwen(str(path)),
            instrucao=INSTRUCAO,
            tipo="nome_registrado",
        )
        gasto = time.perf_counter() - ini
        total += gasto
        print(f"{termo}: {resultado.texto_bruto.strip()} | {gasto:.1f} s", flush=True)
    print(f"MEDIA {total / len(TERMOS):.1f} s | max_pixels={args.max_pixels}")


if __name__ == "__main__":
    main()
