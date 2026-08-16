from __future__ import annotations

"""Executa de forma retomavel a fila de nomes de um livro."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Settings, default_config_path
from src.database.connection import Database
from src.database.repository import Repository
from src.services.name_processing import NameBatchRunner


def _parse_terms(value: str) -> list[int]:
    terms: set[int] = set()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item.strip()) for item in part.split("-", 1))
            terms.update(range(min(start, end), max(start, end) + 1))
        else:
            terms.add(int(part))
    return sorted(terms)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Processa nomes manuscritos com fila persistente"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(r"C:\ProgramData\DigitalizadorLivros\digitalizador.db"),
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--livro", default="A-07")
    parser.add_argument("--limite-qwen", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reprocessar-termos", default="",
        help="termos separados por virgula ou intervalo, por exemplo 6802-6811",
    )
    args = parser.parse_args()

    db = Database(args.db)
    db.connect()
    try:
        livro = db.fetchone(
            "SELECT id,codigo FROM livro WHERE upper(codigo)=upper(?)",
            (args.livro,),
        )
        if not livro:
            raise RuntimeError(f"livro {args.livro!r} nao encontrado")
        repo = Repository(db)
        lote = repo.criar_ou_sincronizar_lote_nomes(int(livro["id"]))
        lote_id = int(lote["id"])
        terms = _parse_terms(args.reprocessar_termos)
        if terms:
            reopened = repo.reabrir_qwen_para_termos(
                int(livro["id"]), terms,
                motivo="novo localizador posicional por rótulo impresso",
            )
            print(json.dumps({"reabertos": reopened, "termos": terms}), flush=True)
    finally:
        db.close()

    def progress(summary: dict, label: str) -> None:
        counts = summary.get("contagens") or {}
        print(json.dumps({
            "evento": label,
            "lote": lote_id,
            "status": summary.get("status"),
            "qwen_pendente": counts.get("qwen_nome:pendente", 0),
            "qwen_revisar": counts.get("qwen_nome:revisar", 0),
            "qwen_sem_resultado": counts.get("qwen_nome:sem_resultado", 0),
            "qwen_falhou": counts.get("qwen_nome:falhou", 0),
        }, ensure_ascii=False), flush=True)

    result = NameBatchRunner(
        db_path=args.db,
        settings=Settings(args.config),
        lote_id=lote_id,
        max_workers=args.workers,
        max_qwen_items=args.limite_qwen,
        on_progress=progress,
    ).run()
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
