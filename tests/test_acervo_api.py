from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import Database
from src.database.repository import Repository
from src.services.acervo_api import AcervoApiServer


def test_api_entrega_json_e_foto_com_caminho_acentuado(tmp_path):
    db = Database(tmp_path / "api.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6,
        tipo_id=1,
        codigo="A-07",
        nome_capa="Nascimentos",
        total_folhas=1,
        primeira_folha=1,
        ultima_folha=1,
        frente_verso=1,
        registros_por_face=1,
        termo_inicial=6801,
        termo_final=6801,
    )
    pasta = tmp_path / "pasta com acentuação"
    pasta.mkdir()
    foto = pasta / "registro ç.jpg"
    imagem = np.full((100, 80, 3), 230, dtype=np.uint8)
    sucesso, codificado = cv2.imencode(".jpg", imagem)
    assert sucesso
    foto.write_bytes(codificado.tobytes())
    imagem_id = repo.registrar_imagem(
        livro_id=livro_id,
        ordem_captura=1,
        caminho_original=str(foto),
        caminho_thumb=str(foto),
        folha_estimada=1,
        face="frente",
        termo_inicial=6801,
        termo_final=6801,
        duplicidade_status="unico",
    )
    registro = repo.sincronizar_registros_imagem(imagem_id)[0]
    repo.salvar_metadado_tratado(
        imagem_id=imagem_id,
        registro_id=registro["id"],
        tipo="nome_registrado",
        valor="Nome Sugerido",
        confianca=0.45,
        fonte="ocr_nome_rapido",
        motor="rapidocr",
        status="precisa_revisao",
    )
    server = AcervoApiServer(repo, port=0)
    server.start()
    try:
        with urllib.request.urlopen(
            f"{server.base_url}/api/v1/registros/{registro['id']}", timeout=3
        ) as resposta:
            item = json.loads(resposta.read())
        assert item["livro_codigo"] == "A-07"
        assert item["foto_url"].endswith(f"/imagens/{imagem_id}")
        assert item["imagem_url"] == item["foto_url"]
        assert item["caminho_imagem"] == str(foto)
        assert item["imagem"]["path"] == str(foto)
        assert item["imagem"]["nome"] == foto.name
        assert item["imagem"]["tamanho_bytes"] == foto.stat().st_size
        assert item["nome_confirmado"] == ""
        assert item["nome_sugerido"] == "Nome Sugerido"
        assert item["nome_status"] == "precisa_revisao"
        assert item["nome_confianca"] == 0.45
        assert item["nome_eh_confirmado"] is False
        with urllib.request.urlopen(item["foto_url"], timeout=3) as resposta:
            assert resposta.status == 200
            assert resposta.headers["Access-Control-Allow-Origin"] == "*"
            assert resposta.read() == foto.read_bytes()
    finally:
        server.stop()
        db.close()
