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
from src.ocr.base import OCRResult
from src.ocr.combiner import OCRCombiner
from src.services.scan_pipeline import ScanPipeline
from src.session.scan_session import ScanSession
from src.ui.term_search_dialog import posicao_do_termo


def _resultado(texto: str, motor: str = "htr") -> list[OCRResult]:
    return [OCRResult(texto_bruto=texto, motor=motor)]


def test_extrai_termo_com_separador_de_milhar():
    termo = OCRCombiner().extrair_termo(_resultado("Número 6.801"), 6801, 6802)

    assert termo.valor == 6801
    assert termo.status == "confirmado"


def test_leitura_parcial_nao_substitui_sequencia():
    termo = OCRCombiner().extrair_termo(
        _resultado("Número 804\nde mil novecentos e oitenta e oito"),
        6801,
        6802,
        fallback_sequencia=True,
    )

    assert termo.valor == 6801
    assert termo.status == "inferido_sequencia"
    assert termo.motor_principal == "sequencia"


def test_numero_completo_fora_da_ordem_vai_para_revisao():
    termo = OCRCombiner().extrair_termo(
        _resultado("Número 7197"),
        6801,
        6802,
        fallback_sequencia=True,
    )

    assert termo.valor == 6801
    assert termo.status == "precisa_revisao"
    assert termo.alternativas[0]["valor"] == 7197


def test_intervalo_inclusivo_e_worker_independente_da_sessao(tmp_path, monkeypatch):
    db = Database(tmp_path / "teste.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6,
        tipo_id=1,
        codigo="A-07",
        total_folhas=300,
        primeira_folha=1,
        ultima_folha=300,
        frente_verso=1,
        registros_por_face=2,
        termo_inicial=6801,
        termo_final=8000,
    )
    session = ScanSession(repo)
    session.selecionar_livro(livro_id)

    assert session.intervalo_termos_atual == (6801, 6802)

    monkeypatch.setattr(ScanPipeline, "_init_ocr", lambda self: None)
    pipeline = ScanPipeline(repo, session, tmp_path / "acervo")
    imagem = np.random.default_rng(7).integers(0, 256, (220, 180, 3), dtype=np.uint8)
    caminho = tmp_path / "pagina.jpg"
    assert cv2.imwrite(str(caminho), imagem)

    resultado = pipeline.processar_imagem_imediato(str(caminho))
    gravada = repo.get_imagem(resultado["imagem_id"])

    assert (resultado["termo_inicial"], resultado["termo_final"]) == (6801, 6802)
    assert (gravada["termo_inicial"], gravada["termo_final"]) == (6801, 6802)
    assert gravada["dhash"]
    assert repo.buscar_imagem_por_termo(livro_id, 6801)["id"] == gravada["id"]
    assert repo.buscar_imagem_por_termo(livro_id, 6802)["id"] == gravada["id"]
    assert repo.buscar_imagem_por_termo(livro_id, 6803) is None
    assert session.intervalo_termos_atual == (6803, 6804)

    repetida = pipeline.processar_imagem_imediato(str(caminho))
    repetida_db = repo.get_imagem(repetida["imagem_id"])
    assert repetida["duplicidade"]["status"] == "duplicata_confirmada"
    assert repetida["sessao_avancou"] is False
    assert (repetida_db["termo_inicial"], repetida_db["termo_final"]) == (6801, 6802)
    assert session.intervalo_termos_atual == (6803, 6804)

    sessao_reaberta = ScanSession(repo)
    sessao_reaberta.selecionar_livro(livro_id)
    assert sessao_reaberta.intervalo_termos_atual == (6803, 6804)
    db.close()


def test_documento_sem_formulario_nao_recebe_termo_nem_avanca(tmp_path, monkeypatch):
    db = Database(tmp_path / "teste.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6,
        tipo_id=1,
        codigo="A-07",
        total_folhas=300,
        primeira_folha=1,
        ultima_folha=300,
        frente_verso=1,
        registros_por_face=2,
        termo_inicial=6801,
        termo_final=8000,
    )
    session = ScanSession(repo)
    session.selecionar_livro(livro_id)

    monkeypatch.setattr(ScanPipeline, "_init_ocr", lambda self: None)
    pipeline = ScanPipeline(repo, session, tmp_path / "acervo")
    monkeypatch.setattr(
        pipeline,
        "_classificar_documento",
        lambda _image: {
            "tipo": "documento_nao_registro",
            "rotacao": 0,
            "confianca": 1.0,
            "motivo": "estrutura de dois registros nao encontrada",
            "texto": "TERMO DE ABERTURA",
        },
    )

    imagem = np.full((720, 960, 3), 235, np.uint8)
    caminho = tmp_path / "abertura.jpg"
    assert cv2.imwrite(str(caminho), imagem)

    resultado = pipeline.processar_imagem_imediato(str(caminho))
    gravada = repo.get_imagem(resultado["imagem_id"])

    assert resultado["tipo_documento"] == "documento_nao_registro"
    assert resultado["sessao_avancou"] is False
    assert (resultado["termo_inicial"], resultado["termo_final"]) == (None, None)
    assert (gravada["termo_inicial"], gravada["termo_final"]) == (None, None)
    assert gravada["tipo_documento"] == "documento_nao_registro"
    assert session.intervalo_termos_atual == (6801, 6802)
    assert "classificar_documento" in {
        revisao["tipo"] for revisao in repo.listar_revisoes_pendentes()
    }
    db.close()


def test_foto_com_mao_entra_na_fila_sem_travar_sequencia(tmp_path, monkeypatch):
    db = Database(tmp_path / "teste.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6, tipo_id=1, codigo="A-07", total_folhas=300,
        primeira_folha=1, ultima_folha=300, frente_verso=1,
        registros_por_face=2, termo_inicial=6801, termo_final=8000,
    )
    session = ScanSession(repo)
    session.selecionar_livro(livro_id)
    monkeypatch.setattr(ScanPipeline, "_init_ocr", lambda self: None)
    pipeline = ScanPipeline(repo, session, tmp_path / "acervo")

    imagem = np.full((720, 960, 3), 235, np.uint8)
    for y in range(80, 680, 28):
        cv2.line(imagem, (80, y), (900, y), (50, 50, 50), 2)
    cv2.rectangle(imagem, (10, 10), (440, 700), (85, 135, 205), -1)
    caminho = tmp_path / "pagina_com_mao.jpg"
    assert cv2.imwrite(str(caminho), imagem)

    resultado = pipeline.processar_imagem_imediato(str(caminho))
    pendentes = repo.listar_revisoes_pendentes()

    assert resultado["qualidade"]["repetir_captura"] is True
    assert resultado["sessao_avancou"] is True
    assert session.intervalo_termos_atual == (6803, 6804)
    assert pendentes[0]["tipo"] == "refazer_captura"
    assert pendentes[0]["folha_estimada"] == 1
    assert pendentes[0]["face"] == "frente"
    assert (pendentes[0]["termo_inicial"], pendentes[0]["termo_final"]) == (6801, 6802)
    assert "mao ou objeto" in pendentes[0]["detalhes"]

    limpa = np.full((720, 960, 3), 235, np.uint8)
    for y in range(80, 680, 28):
        cv2.line(limpa, (80, y), (900, y), (50, 50, 50), 2)
    refoto = tmp_path / "pagina_corrigida.jpg"
    assert cv2.imwrite(str(refoto), limpa)
    repo.criar_execucao_ocr(
        imagem_id=resultado["imagem_id"],
        registro_id=None,
        motor="pipeline-ocr-v1",
        texto_bruto="resultado da foto antiga",
    )
    substituida = pipeline.substituir_captura(
        resultado["imagem_id"], pendentes[0]["id"], str(refoto)
    )
    gravada = repo.get_imagem(resultado["imagem_id"])

    assert substituida["substituida"] is True
    assert gravada["caminho_original"] == str(refoto)
    assert (gravada["termo_inicial"], gravada["termo_final"]) == (6801, 6802)
    assert session.intervalo_termos_atual == (6803, 6804)
    assert repo.listar_revisoes_pendentes() == []
    assert gravada["precisa_revisao"] == 0
    assert substituida["ocr_pendente"] is True
    assert repo.get_execucao_ocr_ativa(
        imagem_id=resultado["imagem_id"],
        registro_id=None,
        motor="pipeline-ocr-v1",
    ) is None
    db.close()


def test_posicao_do_termo_em_face_com_dois_registros():
    assert posicao_do_termo(6801, 6801, 6802) == (0, 2)
    assert posicao_do_termo(6802, 6801, 6802) == (1, 2)
