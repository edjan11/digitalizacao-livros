from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import Database
from src.database.repository import Repository
from src.imaging.document import retificar_formulario
from src.imaging.record_regions import bbox_linha_nome, recortar_bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--inicio", type=int, default=6801)
    parser.add_argument("--quantidade", type=int, default=20)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--face")
    args = parser.parse_args()
    db = Database(args.db)
    db.connect()
    try:
        filtro_face = "AND r.face=?" if args.face else ""
        parametros = (
            (args.inicio, args.face, args.quantidade)
            if args.face
            else (args.inicio, args.quantidade)
        )
        rows = db.fetchall(
            f"""
            SELECT r.id, r.termo, r.indice_na_imagem, r.face,
                   i.caminho_original, i.rotacao_visualizacao
            FROM registro r JOIN imagem i ON i.id=r.imagem_id
            WHERE r.livro_id=1 AND r.termo>=? {filtro_face}
            ORDER BY r.termo LIMIT ?
            """,
            parametros,
        )
    finally:
        db.close()
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    linhas = []
    imagens = []
    for row in rows:
        caminho = Path(row["caminho_original"])
        dados = np.fromfile(str(caminho), dtype=np.uint8)
        imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
        rotacao = int(row.get("rotacao_visualizacao") or 0)
        if rotacao == 180:
            imagem = cv2.rotate(imagem, cv2.ROTATE_180)
        elif rotacao == 90:
            imagem = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
        elif rotacao == 270:
            imagem = cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)
        retificada = retificar_formulario(imagem).image
        crop = recortar_bbox(
            retificada,
            bbox_linha_nome(int(row["indice_na_imagem"]), 2),
        )
        linhas.append(
            f"{row['termo']} | idx={row['indice_na_imagem']} | {caminho.name}"
        )
        imagens.append(crop)
    largura = 900
    altura = 90
    cols = 2
    rows_n = (len(imagens) + cols - 1) // cols
    tela = np.full((rows_n * (altura + 28), cols * largura, 3), 245, np.uint8)
    for idx, (crop, legenda) in enumerate(zip(imagens, linhas)):
        thumb = cv2.resize(crop, (largura, altura), interpolation=cv2.INTER_AREA)
        x = (idx % cols) * largura
        y = (idx // cols) * (altura + 28)
        tela[y:y + altura, x:x + largura] = thumb
        cv2.putText(
            tela,
            legenda[:105],
            (x + 4, y + altura + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(args.saida.parent / f"{idx:02d}_{int(rows[idx]['termo'])}.jpg"), crop)
    cv2.imwrite(str(args.saida), tela)
    print("\n".join(f"{idx}: {legenda}" for idx, legenda in enumerate(linhas)))
    print(f"MONTAGEM={args.saida}")


if __name__ == "__main__":
    main()
