from __future__ import annotations

"""Audita associações Qwen e prepara a fila persistente de um livro.

O comando é deliberadamente explícito: sem ``--apply`` ele apenas mostra o
estado atual. As fotografias nunca são abertas para escrita.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import Database
from src.database.repository import Repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--livro", default="A-07")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    db = Database(args.db)
    db.connect()
    repo = Repository(db)
    try:
        livros = db.fetchall(
            """
            SELECT l.id, l.codigo, a.nome AS acervo, o.nome AS oficio,
                   COUNT(DISTINCT r.id) AS registros
            FROM livro l
            LEFT JOIN acervo a ON a.id=l.acervo_id
            LEFT JOIN oficio o ON o.id=l.oficio_id
            LEFT JOIN registro r ON r.livro_id=l.id
            WHERE upper(l.codigo)=upper(?)
            GROUP BY l.id, l.codigo, a.nome, o.nome
            """,
            (args.livro,),
        )
        if len(livros) != 1:
            raise RuntimeError(
                f"esperado exatamente um livro {args.livro!r}; encontrados: {livros}"
            )
        livro = livros[0]
        livro_id = int(livro["id"])
        saida: dict = {
            "aplicado": bool(args.apply),
            "banco": str(args.db.resolve()),
            "livro": livro,
            "qwen_incompativeis_ativos_antes": len(
                repo.listar_associacoes_qwen_invalidas(livro_id=livro_id)
            ),
        }
        if args.apply:
            saida["qwen_descartados"] = repo.auditar_associacoes_qwen(
                livro_id=livro_id
            )
            saida["sugestoes_rapidas_rebaixadas"] = (
                repo.rebaixar_sugestoes_rapidas_antigas(livro_id=livro_id)
            )
            lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)
            saida["lote"] = repo.resumo_processamento(int(lote["id"]))
        saida["termo_6802"] = repo.buscar_registros(
            termo=6802, livro_id=livro_id, limite=10
        )
        saida["qwen_incompativeis_ativos_depois"] = len(
            repo.listar_associacoes_qwen_invalidas(livro_id=livro_id)
        )
        print(json.dumps(saida, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
