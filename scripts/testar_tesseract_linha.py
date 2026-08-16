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
from src.ocr.tesseract_engine import TesseractEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foto", type=Path, required=True)
    parser.add_argument("--indice", type=int, default=0)
    args = parser.parse_args()
    dados = np.fromfile(str(args.foto), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    faixa = recortar_bbox(retificar_formulario(imagem).image, bbox_faixa_nome(args.indice, 2))
    linha = faixa[int(faixa.shape[0] * 0.54):int(faixa.shape[0] * 0.88)]
    engine = TesseractEngine(lang="por")
    for psm in (6, 7, 13):
        print(f"PSM {psm}: {engine.read_array(linha, psm=psm).strip()!r}")


if __name__ == "__main__":
    main()
