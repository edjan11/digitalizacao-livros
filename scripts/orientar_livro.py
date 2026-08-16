from __future__ import annotations

"""Audita/aplica orientacao logica por coorte sem alterar os JPGs."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import Database
from src.database.repository import Repository
from src.services.orientation_processing import OrientationBatchRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Orienta fotos de um livro por coorte")
    parser.add_argument("--db", type=Path, default=Path(r"C:\ProgramData\DigitalizadorLivros\digitalizador.db"))
    parser.add_argument("--livro", required=True)
    parser.add_argument("--amostras", type=int, default=5)
    parser.add_argument("--intervalo", type=int, default=25)
    parser.add_argument("--aplicar", action="store_true")
    args = parser.parse_args()

    db = Database(args.db)
    db.connect()
    try:
        result = OrientationBatchRunner(
            Repository(db), sample_count=args.amostras,
            validation_interval=args.intervalo,
            on_progress=lambda message: print(message, flush=True),
        ).run(args.livro, apply=args.aplicar)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if all(group["approved"] for group in result["groups"]) else 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
