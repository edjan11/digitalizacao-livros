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
from src.ocr.got_ocr_engine import GOTOCRProvider
from src.ocr.tesseract_engine import TesseractEngine


def abrir(caminho: Path):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise RuntimeError(f"não abriu: {caminho}")
    return retificar_formulario(imagem).image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foto", type=Path, required=True)
    parser.add_argument("--indice", type=int, default=0)
    parser.add_argument("--y1", type=int, default=0)
    parser.add_argument("--y2", type=int, default=0)
    parser.add_argument("--x1", type=int, default=0)
    parser.add_argument("--x2", type=int, default=0)
    parser.add_argument("--got", type=Path, default=ROOT / "models" / "got-ocr-2.0-hf")
    args = parser.parse_args()
    faixa = recortar_bbox(abrir(args.foto), bbox_faixa_nome(args.indice, 2))
    if args.y2 > args.y1:
        faixa = faixa[max(0, args.y1):min(faixa.shape[0], args.y2)]
    if args.x2 > args.x1:
        faixa = faixa[:, max(0, args.x1):min(faixa.shape[1], args.x2)]
    print(f"foto={args.foto.name} recorte={faixa.shape[1]}x{faixa.shape[0]}")
    got = GOTOCRProvider(args.got, permitir_download=False, max_new_tokens=128)
    resultado = got.recognize(faixa, fast=False)
    print(f"got_ms={resultado.tempo_ms:.0f}\nGOT:\n{resultado.texto_bruto}")


if __name__ == "__main__":
    main()
