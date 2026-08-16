from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--inicio", type=int, default=6801)
    parser.add_argument("--fim", type=int, default=6811)
    parser.add_argument("--resumo", action="store_true")
    args = parser.parse_args()
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    if args.resumo:
        for row in con.execute(
            "SELECT etapa,status,COUNT(*) AS n FROM processamento_item "
            "WHERE lote_id=1 GROUP BY etapa,status ORDER BY etapa,status"
        ):
            print(dict(row))
        con.close()
        return
    rows = con.execute(
        """
        SELECT r.termo, p.status, p.motor, p.resultado, p.confianca,
               p.tempo_ms, o.texto_bruto
        FROM processamento_item p
        JOIN registro r ON r.id=p.registro_id
        LEFT JOIN ocr_execucao o ON o.registro_id=r.id
          AND o.motor='ocr-nomes-rapido-v2'
        WHERE p.lote_id=1 AND p.etapa='ocr_nome_rapido'
          AND r.termo BETWEEN ? AND ?
        ORDER BY r.termo
        """,
        (args.inicio, args.fim),
    ).fetchall()
    for row in rows:
        print(dict(row))
    con.close()


if __name__ == "__main__":
    main()
