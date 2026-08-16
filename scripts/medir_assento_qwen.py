"""Mede a leitura estruturada completa do assento (caminho de producao).

Usa os recortes de registro inteiro (.tmp_qwen_20) com QwenRecordAnalyzer,
que e o mesmo analisador do worker de nomes (nome + termo + mae + data).
Compara dois limites de pixels sem alterar nenhum arquivo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.qwen_vl_engine import QwenRecordAnalyzer

TERMOS = ["6801", "6802", "6805", "6809", "6810"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".tmp_qwen_20")
    ap.add_argument("--max-pixels", type=int, default=301056)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    analisador = QwenRecordAnalyzer(
        max_new_tokens=128,
        min_pixels=128 * 28 * 28,
        max_pixels=args.max_pixels,
        threads=args.threads,
    )
    total = 0.0
    for termo in TERMOS:
        path = next(Path(args.input).glob(f"*_{termo}.jpg"))
        ini = time.perf_counter()
        campos, resultado = analisador.analisar_registro(str(path))
        gasto = time.perf_counter() - ini
        total += gasto
        nome = str(campos.get("nome_registrado") or "").strip()
        print(
            f"{termo}: {nome} | {campos.get('termo')} | {campos.get('nome_mae')} | "
            f"{campos.get('data_registro')} | {gasto:.1f} s",
            flush=True,
        )
    print(f"MEDIA {total / len(TERMOS):.1f} s | max_pixels={args.max_pixels}")


if __name__ == "__main__":
    main()
