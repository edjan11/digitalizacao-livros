"""Teste conservador de OCR nas faces auditadas do Livro A-07.

O OCR nunca define a numeracao neste script. Ele apenas confere o termo que ja
foi determinado pela sequencia fisica auditada. A abertura, os indices e as
dez recapturas rejeitadas nao entram como registros.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.combiner import OCRCombiner
from src.ocr.engines import RapidOCRProvider, TesseractProvider
from src.ocr.htr_engine import HTREngine
from src.services.organized_book_importer import A07_KEEP, auditar_a07, _read_image


def _faces(audit, face: str):
    if face == "frente":
        for folha, path in enumerate(audit.frentes, 1):
            start = 6801 + (folha - 1) * 4
            yield folha, path, 0, (start, start + 1)
        return
    for position, path in enumerate(audit.versos, 1):
        if position in audit.posicoes_rejeitadas:
            continue
        terms = audit.termos_por_posicao_verso[position]
        folha = ((terms[0] - 6803) // 4) + 1
        yield folha, path, 180 if position == 1 else 0, terms


def testar_livro(root: Path, face: str, max_imagens: int, usar_htr: bool) -> None:
    audit = auditar_a07(root)
    faces = list(_faces(audit, face))
    if max_imagens > 0:
        faces = faces[:max_imagens]

    combiner = OCRCombiner()
    rapid = RapidOCRProvider(apenas_cabecalhos=True)
    if rapid.is_available():
        combiner.add_provider(rapid)
    tesseract = TesseractProvider()
    if tesseract.is_available():
        combiner.add_provider(tesseract)
    if usar_htr:
        htr = HTREngine()
        htr.load()
        combiner.add_provider(htr)

    print("Livro A-07 — teste somente em faces de registro auditadas")
    print(f"Face: {face} | imagens: {len(faces)} | motores: {[p.name for p in combiner.providers]}")
    print("O termo confirmado vem da sequencia; OCR e apenas conferencia.\n")

    concordancias = 0
    for index, (folha, path, rotation, expected) in enumerate(faces, 1):
        image = _read_image(path)
        if rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        results = combiner.recognize_all(image, fast=True)
        check = combiner.extrair_termo(
            results,
            esperado_min=expected[0],
            esperado_max=expected[1],
            fallback_sequencia=False,
        )
        read_value = check.valor
        agreed = read_value is not None and expected[0] <= read_value <= expected[1]
        concordancias += int(agreed)
        raw = " | ".join(
            f"{result.motor}: {(result.texto_bruto or '')[:90].replace(chr(10), ' / ')}"
            for result in results
        )
        print(
            f"[{index:03d}] folha {folha:03d} {path.name} | "
            f"CONFIRMADO {expected[0]}-{expected[1]} | "
            f"OCR {read_value or '?'} ({check.status}, {check.confianca:.2f}) | "
            f"{'CONFERE' if agreed else 'NAO CONFERE — revisar OCR, nao renumerar'}"
        )
        print(f"      {raw}")

    print(
        f"\nResultado: {concordancias}/{len(faces)} conferencias OCR. "
        "Falha do OCR nao altera folha nem termo."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(r"D:\A - 07"))
    parser.add_argument("--face", choices=("frente", "verso"), default="frente")
    parser.add_argument("--max", type=int, default=10, dest="max_imagens")
    parser.add_argument(
        "--htr",
        action="store_true",
        help="Inclui EasyOCR/HTR (lento). Sem esta opção, usa apenas motores rápidos.",
    )
    args = parser.parse_args()
    testar_livro(args.root, args.face, args.max_imagens, args.htr)


if __name__ == "__main__":
    main()
