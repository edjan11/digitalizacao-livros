from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import data_dir
from src.database.connection import Database
from src.database.repository import Repository
from src.services.generic_book_importer import (
    BookImportSpec,
    GenericBookImporter,
    auditar_livro,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita/importa o Livro A-16 sem alterar originais")
    parser.add_argument("--root", default=r"D:\A - 16")
    parser.add_argument("--importar", action="store_true")
    parser.add_argument(
        "--sem-derivadas", action="store_true",
        help="cadastra termos/imagens agora e deixa orientacao para o worker retomavel",
    )
    args = parser.parse_args()
    root = Path(args.root)
    report_dir = data_dir() / "auditorias" / "A-16"
    report_dir.mkdir(parents=True, exist_ok=True)

    def progress(current: int, total: int, label: str) -> None:
        print(f"[{current:03}/{total:03}] {label}", flush=True)

    audit = auditar_livro(root, BookImportSpec.a16(), on_progress=progress)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = report_dir / f"auditoria-a16-{stamp}.json"
    report.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RELATORIO={report}")
    print(
        f"registros={len(audit.registros)} indices={len(audit.indices)} "
        f"faltantes={len(audit.faltantes)} duplicados={len(audit.duplicados)} "
        f"nao_resolvidos={len(audit.nao_resolvidos)}"
    )
    if not args.importar:
        return 0 if audit.ready else 2
    if not audit.ready:
        print("IMPORTACAO_CANCELADA=auditoria_inconclusiva")
        return 2
    db = Database()
    try:
        db.connect()
        result = GenericBookImporter(
            Repository(db), normalized_root=data_dir() / "normalizadas"
        ).importar(
            audit, create_derivatives=not args.sem_derivadas, on_progress=progress
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
