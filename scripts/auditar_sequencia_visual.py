"""Gera folhas de contato dos cabecalhos para auditar a sequencia de um livro.

O arquivo original nunca e alterado. A saida mostra a posicao cronologica,
o nome da foto e as duas celulas "Numero" da face, lado a lado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _cabecalhos(path: Path, girar_180: bool) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if girar_180:
            image = image.rotate(180, expand=False)
        width, height = image.size
        x1, x2 = int(width * 0.08), int(width * 0.37)
        boxes = (
            (x1, int(height * 0.015), x2, int(height * 0.19)),
            (x1, int(height * 0.425), x2, int(height * 0.625)),
        )
        crops = [image.crop(box) for box in boxes]

    target_width = 330
    resized = []
    for crop in crops:
        ratio = target_width / crop.width
        resized.append(
            crop.resize(
                (target_width, max(1, int(crop.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        )
    height = max(item.height for item in resized)
    result = Image.new("RGB", (target_width * 2, height), "white")
    result.paste(resized[0], (0, 0))
    result.paste(resized[1], (target_width, 0))
    return result


def gerar(
    folder: Path,
    output: Path,
    inicio: int = 1,
    fim: int | None = None,
    por_pagina: int = 16,
    girar_posicoes: set[int] | None = None,
) -> list[Path]:
    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    fim = min(fim or len(files), len(files))
    selected = list(enumerate(files[inicio - 1:fim], start=inicio))
    output.mkdir(parents=True, exist_ok=True)
    label_font = _font(17)
    pages = []
    girar_posicoes = girar_posicoes or set()

    columns = 2
    card_width, card_height = 680, 300
    for page_index in range(0, len(selected), por_pagina):
        chunk = selected[page_index:page_index + por_pagina]
        rows = (len(chunk) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * card_width, rows * card_height), "#202124")
        draw = ImageDraw.Draw(canvas)
        for local_index, (position, path) in enumerate(chunk):
            column = local_index % columns
            row = local_index // columns
            left, top = column * card_width, row * card_height
            label = f"{position:03d}  {path.name}"
            draw.text((left + 8, top + 6), label, font=label_font, fill="white")
            crop = _cabecalhos(path, position in girar_posicoes)
            crop.thumbnail((card_width - 16, card_height - 38), Image.Resampling.LANCZOS)
            canvas.paste(crop, (left + 8, top + 34))
        page_path = output / f"cabecalhos_{chunk[0][0]:03d}_{chunk[-1][0]:03d}.jpg"
        canvas.save(page_path, quality=92)
        pages.append(page_path)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".data/auditoria_sequencia"))
    parser.add_argument("--inicio", type=int, default=1)
    parser.add_argument("--fim", type=int)
    parser.add_argument("--por-pagina", type=int, default=16)
    parser.add_argument(
        "--girar-180",
        default="",
        help="Posicoes separadas por virgula que devem ser giradas, por exemplo: 1,7",
    )
    args = parser.parse_args()
    rotations = {int(value) for value in args.girar_180.split(",") if value.strip()}
    for page in gerar(
        args.folder,
        args.output,
        inicio=args.inicio,
        fim=args.fim,
        por_pagina=args.por_pagina,
        girar_posicoes=rotations,
    ):
        print(page)


if __name__ == "__main__":
    main()
