from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import Database
from src.database.repository import Repository
from src.metadata.extractor import extrair_metadados
from src.ocr.base import OCRResult, OCRToken
from src.ocr.got_ocr_engine import _tokens_do_texto
from src.services.scan_pipeline import ScanPipeline
from src.session.scan_session import ScanSession


def _acervo_com_imagem(tmp_path):
    db = Database(tmp_path / "consulta.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6, tipo_id=1, codigo="A-07", nome_capa="Nascimentos",
        total_folhas=300, primeira_folha=1, ultima_folha=300,
        frente_verso=1, registros_por_face=2,
        termo_inicial=6801, termo_final=8000,
    )
    path = tmp_path / "pagina.jpg"
    image = np.full((600, 400, 3), 235, np.uint8)
    for y in range(40, 580, 25):
        cv2.line(image, (20, y), (380, y), (50, 50, 50), 1)
    assert cv2.imwrite(str(path), image)
    imagem_id = repo.registrar_imagem(
        livro_id=livro_id, ordem_captura=1,
        caminho_original=str(path), caminho_thumb=str(path),
        folha_estimada=1, face="frente", termo_inicial=6801,
        termo_final=6802, duplicidade_status="unico",
    )
    registros = repo.sincronizar_registros_imagem(imagem_id)
    return db, repo, livro_id, imagem_id, registros


def test_extrai_nome_e_preserva_tokens_com_caixa():
    resultado = OCRResult(
        motor="htr",
        texto_bruto=(
            "Número 6801\nque recebeu o nome de João Carlos da Silva\n"
            "filho de Antônio da Silva e Maria de Souza\n12/08/1988"
        ),
        tokens=[
            OCRToken(
                tipo="texto_linha", valor="João Carlos da Silva",
                confianca=0.82, motor="htr", bbox=[[0.1, 0.2], [0.8, 0.2]],
            )
        ],
    )
    deteccoes = extrair_metadados(resultado)

    nomes = [d for d in deteccoes if d.tipo == "nome_registrado"]
    assert nomes[0].valor_tratado == "João Carlos da Silva"
    assert nomes[0].valor_normalizado == "JOAO CARLOS DA SILVA"
    assert any(d.tipo == "data" and d.valor_normalizado == "12 08 1988" for d in deteccoes)
    assert any(d.bbox_json for d in deteccoes)


def test_metadados_sao_pesquisaveis_por_nome_termo_livro_e_acervo(tmp_path):
    db, repo, livro_id, imagem_id, registros = _acervo_com_imagem(tmp_path)
    resultado = OCRResult(
        motor="htr",
        texto_bruto="que recebeu o nome de João Carlos da Silva",
    )
    execucao = repo.criar_execucao_ocr(
        imagem_id=imagem_id, registro_id=registros[0]["id"], motor="htr",
        texto_bruto=resultado.texto_bruto, tempo_ms=25,
    )
    repo.salvar_deteccoes_ocr(
        execucao_id=execucao, imagem_id=imagem_id,
        registro_id=registros[0]["id"],
        deteccoes=[d.to_dict() for d in extrair_metadados(resultado)],
    )

    por_nome = repo.buscar_registros(texto="joao carlos")
    por_termo = repo.buscar_registros(termo=6802)
    por_livro = repo.buscar_registros(texto="A-07", livro_id=livro_id)
    por_acervo = repo.buscar_registros(acervo_id=6)

    assert len(por_nome) == 1
    assert por_nome[0]["termo"] == 6801
    assert "João Carlos" in por_nome[0]["nomes"]
    assert [r["termo"] for r in por_termo] == [6802]
    assert len(por_livro) == 2
    assert len(por_acervo) == 2
    db.close()


def test_busca_fuzzy_por_erro_de_grafia_nao_confirma_nome(tmp_path):
    db, repo, _, imagem_id, registros = _acervo_com_imagem(tmp_path)
    resultado = OCRResult(
        motor="qwen",
        texto_bruto="que recebeu o nome de Anderson da Silva Cruz",
    )
    execucao = repo.criar_execucao_ocr(
        imagem_id=imagem_id, registro_id=registros[0]["id"], motor="qwen",
        texto_bruto=resultado.texto_bruto, tempo_ms=20,
    )
    repo.salvar_deteccoes_ocr(
        execucao_id=execucao, imagem_id=imagem_id,
        registro_id=registros[0]["id"],
        deteccoes=[d.to_dict() for d in extrair_metadados(resultado)],
    )

    encontrados = repo.buscar_registros(texto="Andersn da Silva Cruz")
    assert encontrados
    assert encontrados[0]["termo"] == 6801
    assert encontrados[0]["busca_fuzzy"] is True
    assert encontrados[0]["nome_busca_similaridade"] > 0.90
    # O campo continua sendo sugestão; a aproximação não muda o status.
    assert encontrados[0]["nome_status"] != "confirmado"
    db.close()


def test_reindexacao_preserva_historico_e_pesquisa_so_versao_ativa(tmp_path):
    db, repo, _, imagem_id, registros = _acervo_com_imagem(tmp_path)
    registro_id = registros[0]["id"]

    primeira = repo.criar_execucao_ocr(
        imagem_id=imagem_id, registro_id=registro_id, motor="htr",
        texto_bruto="nome de José Antigo",
    )
    repo.salvar_deteccoes_ocr(
        execucao_id=primeira, imagem_id=imagem_id, registro_id=registro_id,
        deteccoes=[d.to_dict() for d in extrair_metadados(
            OCRResult(motor="htr", texto_bruto="nome de José Antigo")
        )],
    )
    segunda = repo.criar_execucao_ocr(
        imagem_id=imagem_id, registro_id=registro_id, motor="htr",
        texto_bruto="nome de José Correto",
    )
    repo.salvar_deteccoes_ocr(
        execucao_id=segunda, imagem_id=imagem_id, registro_id=registro_id,
        deteccoes=[d.to_dict() for d in extrair_metadados(
            OCRResult(motor="htr", texto_bruto="nome de José Correto")
        )],
    )

    assert repo.buscar_registros(texto="Jose Antigo") == []
    assert len(repo.buscar_registros(texto="Jose Correto")) == 1
    historico = db.fetchall(
        "SELECT id, ativo FROM ocr_execucao WHERE registro_id=? ORDER BY id",
        (registro_id,),
    )
    assert [r["ativo"] for r in historico] == [0, 1]
    db.close()


def test_pipeline_preserva_texto_e_deteccoes_do_ocr(tmp_path, monkeypatch):
    db, repo, livro_id, imagem_id, _ = _acervo_com_imagem(tmp_path)
    session = ScanSession(repo)
    session.selecionar_livro(livro_id)
    monkeypatch.setattr(ScanPipeline, "_init_ocr", lambda self: None)
    pipeline = ScanPipeline(repo, session, tmp_path / "acervo")
    resultado = OCRResult(
        motor="htr",
        texto_bruto="Número 6801\nque recebeu o nome de Ana Beatriz de Souza",
        tempo_ms=321.0,
    )

    pipeline._persistir_resultados_ocr(imagem_id, [resultado])

    execucoes = db.fetchall(
        "SELECT * FROM ocr_execucao WHERE imagem_id=? AND ativo=1",
        (imagem_id,),
    )
    deteccoes = db.fetchall(
        "SELECT * FROM ocr_deteccao WHERE imagem_id=? AND ativo=1",
        (imagem_id,),
    )
    assert execucoes[0]["texto_bruto"].startswith("Número 6801")
    assert any(d["tipo"] == "nome_registrado" for d in deteccoes)
    assert len(repo.buscar_registros(texto="ana beatriz")) == 2
    db.close()


def test_got_converte_transcricao_em_metadados_conservadores():
    tokens = _tokens_do_texto(
        "Número 6801\nque recebeu o nome de Ana Beatriz\nFolha 12\n1988"
    )

    assert any(t.tipo == "texto_linha" and "Ana Beatriz" in t.valor for t in tokens)
    assert any(t.tipo == "termo" and t.valor == "6801" for t in tokens)
    assert any(t.tipo == "folha" and t.valor == "12" for t in tokens)
    assert all(t.motor == "got-ocr2" for t in tokens)
    assert all(t.bbox is None for t in tokens)


def test_fila_pendente_pode_ser_especifica_por_motor(tmp_path):
    db, repo, livro_id, imagem_id, registros = _acervo_com_imagem(tmp_path)
    for registro in registros:
        repo.criar_execucao_ocr(
            imagem_id=imagem_id,
            registro_id=registro["id"],
            motor="tesseract",
            texto_bruto="texto já indexado",
        )

    assert repo.listar_imagens_para_indexacao(livro_id=livro_id) == []
    pendentes_got = repo.listar_imagens_para_indexacao(
        livro_id=livro_id,
        motores_pendentes=["got-ocr2"],
    )
    assert [item["id"] for item in pendentes_got] == [imagem_id]

    for registro in registros:
        repo.criar_execucao_ocr(
            imagem_id=imagem_id,
            registro_id=registro["id"],
            motor="got-ocr2",
            texto_bruto="transcrição GOT",
        )
    assert repo.listar_imagens_para_indexacao(
        livro_id=livro_id,
        motores_pendentes=["got-ocr2"],
    ) == []
    db.close()


def test_ocr_da_fotografia_e_executado_uma_vez_e_reutilizado(tmp_path, monkeypatch):
    db, repo, livro_id, imagem_id, _ = _acervo_com_imagem(tmp_path)
    session = ScanSession(repo)
    session.selecionar_livro(livro_id)
    monkeypatch.setattr(ScanPipeline, "_init_ocr", lambda self: None)
    pipeline = ScanPipeline(repo, session, tmp_path / "acervo")

    class ProviderContado:
        name = "ocr-teste"

        def __init__(self):
            self.chamadas = 0

        def is_available(self):
            return True

        def recognize(self, _image, fast=False):
            self.chamadas += 1
            return OCRResult(
                motor=self.name,
                texto_bruto="Numero 6801",
                tempo_ms=12.0,
            )

    provider = ProviderContado()
    pipeline.combiner.providers = [provider]

    primeira = pipeline.processar_ocr_secundario(imagem_id)
    segunda = pipeline.processar_ocr_secundario(imagem_id)

    assert provider.chamadas == 1
    assert primeira["reutilizado"] is False
    assert segunda["reutilizado"] is True
    assert segunda["mensagem"] == "OCR já processado; resultado carregado do banco"
    execucoes = db.fetchall(
        "SELECT motor, ativo FROM ocr_execucao WHERE imagem_id=? ORDER BY id",
        (imagem_id,),
    )
    assert [(e["motor"], e["ativo"]) for e in execucoes] == [
        ("ocr-teste", 1),
        ("pipeline-ocr-v1", 1),
    ]
    db.close()


def test_execucao_de_pagina_ou_falha_salva_nao_volta_para_fila(tmp_path):
    db, repo, livro_id, imagem_id, registros = _acervo_com_imagem(tmp_path)
    repo.criar_execucao_ocr(
        imagem_id=imagem_id,
        registro_id=None,
        motor="tesseract",
        texto_bruto="OCR completo da fotografia",
    )

    assert repo.listar_imagens_para_indexacao(livro_id=livro_id) == []
    assert repo.listar_imagens_para_indexacao(
        livro_id=livro_id,
        motores_pendentes=["tesseract"],
    ) == []

    repo.invalidar_ocr_imagem(imagem_id)
    for registro in registros:
        repo.criar_execucao_ocr(
            imagem_id=imagem_id,
            registro_id=registro["id"],
            motor="tesseract",
            texto_bruto="",
            sucesso=False,
            erro="teste de falha persistida",
        )
    assert repo.listar_imagens_para_indexacao(livro_id=livro_id) == []
    db.close()


def test_areas_qwen_diferentes_ficam_preservadas(tmp_path):
    db, repo, _, imagem_id, registros = _acervo_com_imagem(tmp_path)
    registro_id = registros[0]["id"]
    for texto, bbox in (
        ("Ana da Silva", "[0.1, 0.1, 0.8, 0.2]"),
        ("Maria de Souza", "[0.1, 0.3, 0.8, 0.4]"),
    ):
        execucao = repo.criar_execucao_ocr(
            imagem_id=imagem_id,
            registro_id=registro_id,
            motor="qwen2-vl-2b-area",
            texto_bruto=texto,
            substituir_ativa=False,
        )
        repo.salvar_deteccoes_ocr(
            execucao_id=execucao,
            imagem_id=imagem_id,
            registro_id=registro_id,
            deteccoes=[{
                "tipo": "texto_linha",
                "valor_original": texto,
                "valor_tratado": texto,
                "valor_normalizado": texto.upper(),
                "motor": "qwen2-vl-2b-area",
                "fonte": "qwen_area",
                "bbox_json": bbox,
            }],
        )

    execucoes = repo.listar_execucoes_ocr_ativas(
        imagem_id=imagem_id,
        registro_id=registro_id,
    )
    areas = repo.listar_deteccoes_area(
        imagem_id=imagem_id,
        registro_id=registro_id,
        tipo="texto_linha",
    )
    assert len(execucoes) == 2
    assert {a["valor_tratado"] for a in areas} == {"Ana da Silva", "Maria de Souza"}
    db.close()
