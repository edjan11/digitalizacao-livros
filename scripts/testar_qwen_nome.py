from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imaging.document import retificar_formulario
from src.imaging.record_regions import bbox_faixa_nome, recortar_bbox
from src.ocr.qwen_vl_engine import QwenAreaAnalyzer, preparar_imagem_qwen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foto", type=Path, required=True)
    parser.add_argument("--foto2", type=Path)
    parser.add_argument("--foto3", type=Path)
    parser.add_argument("--indice", type=int, default=0)
    parser.add_argument("--y1", type=int, default=0)
    parser.add_argument("--y2", type=int, default=0)
    parser.add_argument("--x1", type=int, default=0)
    parser.add_argument("--x2", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    def preparar(caminho: Path):
        dados = np.fromfile(str(caminho), dtype=np.uint8)
        imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
        geometria = retificar_formulario(imagem)
        faixa = recortar_bbox(geometria.image, bbox_faixa_nome(args.indice, 2))
        if args.y2 > args.y1:
            faixa = faixa[max(0, args.y1):min(faixa.shape[0], args.y2)]
        if args.x2 > args.x1:
            faixa = faixa[:, max(0, args.x1):min(faixa.shape[1], args.x2)]
        return preparar_imagem_qwen(faixa)

    inicio = __import__("time").perf_counter()
    analisador = QwenAreaAnalyzer(
        max_new_tokens=32,
        dtype=args.dtype,
        max_pixels=args.max_pixels,
    )
    instrucao = (
        "A imagem é um recorte de um registro civil brasileiro. "
        "Leia SOMENTE o nome manuscrito que vem depois das palavras "
        "impressas 'que recebeu o nome de'. Ignore todo o resto. "
        "Responda apenas com o nome completo, sem explicação. "
        "Se uma letra estiver ilegível, preserve a dúvida e não invente."
    )
    fotos = [args.foto]
    if args.foto2:
        fotos.append(args.foto2)
    if args.foto3:
        fotos.append(args.foto3)
    for caminho in fotos:
        inicio_item = __import__("time").perf_counter()
        resultado = analisador.analisar(
            preparar(caminho), instrucao=instrucao, tipo="nome_registrado"
        )
        print(
            f"foto={caminho.name} tempo_item_ms={resultado.tempo_ms:.0f} "
            f"total_item_ms={( __import__('time').perf_counter() - inicio_item) * 1000:.0f}"
        )
        print(resultado.texto_bruto)
    print(f"total_ms={( __import__('time').perf_counter() - inicio) * 1000:.0f}")


if __name__ == "__main__":
    main()
