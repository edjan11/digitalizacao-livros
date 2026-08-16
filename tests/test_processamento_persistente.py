from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.consulta.main_window import estado_visual_registro
from src.config.settings import Settings
from src.database.connection import Database
from src.database.repository import Repository
from src.imaging.document import retificar_formulario
from src.imaging.record_regions import (
    bbox_data_registro,
    bbox_corresponde_registro,
    bbox_faixa_nome,
    bbox_linha_nome,
    bbox_registro,
)
from src.services.job_context import QwenJobContext, validar_contexto_qwen
from src.services.name_processing import (
    NameBatchRunner,
    _exclusive_worker_lock,
    _localizar_linhas_nome_pagina,
    _nome_parece_legivel,
)
from src.ocr.base import OCRResult
from tests.test_consulta import _acervo_com_imagem


def test_regioes_superior_inferior_e_averbacao_sao_deterministicas():
    superior = bbox_registro(0, 2)
    inferior = bbox_registro(1, 2)
    nome_inferior = bbox_faixa_nome(1, 2)
    linha_inferior = bbox_linha_nome(1, 2)
    data_inferior = bbox_data_registro(1, 2)

    assert superior == (0.045, 0.004, 0.76, 0.496)
    assert inferior == (0.045, 0.490, 0.76, 0.996)
    assert nome_inferior[0] >= 0.14
    assert nome_inferior[2] == 0.76
    assert linha_inferior[0] == 0.28
    assert linha_inferior[1] > nome_inferior[1]
    assert linha_inferior[3] < nome_inferior[3]
    assert data_inferior[0] >= 0.14
    assert data_inferior[2] == 0.76
    assert data_inferior[1] >= inferior[1]
    assert data_inferior[3] < nome_inferior[1]
    assert bbox_corresponde_registro(superior, 0, 2)
    assert not bbox_corresponde_registro(superior, 1, 2)
    assert not bbox_corresponde_registro((0.045, 0.490, 0.82, 0.996), 1, 2)


def test_contexto_qwen_permanece_no_registro_que_iniciou(tmp_path):
    db, repo, _, imagem_id, registros = _acervo_com_imagem(tmp_path)
    repo.atualizar_imagem(imagem_id, sha256="abc123")
    contexto = QwenJobContext(
        registro_id=registros[0]["id"],
        imagem_id=imagem_id,
        termo=6801,
        indice_na_imagem=0,
        total_na_imagem=2,
        bbox=bbox_registro(0, 2),
        imagem_sha256="abc123",
        tipo="registro",
    )

    # A interface pode selecionar o segundo registro enquanto o worker roda;
    # a validação continua resolvendo o primeiro pelo contexto imutável.
    registro_selecionado_depois = registros[1]
    validado = validar_contexto_qwen(repo, contexto)
    assert registro_selecionado_depois["id"] != contexto.registro_id
    assert validado["registro_id"] == registros[0]["id"]

    contexto_errado = QwenJobContext(
        **{**contexto.to_dict(), "registro_id": registros[1]["id"]}
    )
    assert validar_contexto_qwen(repo, contexto_errado) is None
    db.close()


def test_auditoria_descarta_qwen_preso_ao_assento_errado_sem_apagar_historico(tmp_path):
    db, repo, _, imagem_id, registros = _acervo_com_imagem(tmp_path)
    execucao = repo.criar_execucao_ocr(
        imagem_id=imagem_id,
        registro_id=registros[1]["id"],
        motor="qwen2-vl-2b-area",
        texto_bruto='{"nome_registrado":"Begoíng dos Santos Teho"}',
    )
    repo.salvar_deteccoes_ocr(
        execucao_id=execucao,
        imagem_id=imagem_id,
        registro_id=registros[1]["id"],
        deteccoes=[{
            "tipo": "nome_registrado",
            "valor_original": "Begoíng dos Santos Teho",
            "valor_tratado": "Begoíng dos Santos Teho",
            "valor_normalizado": "BEGOING DOS SANTOS TEHO",
            "confianca": 0.45,
            "motor": "qwen2-vl-2b-area",
            "fonte": "qwen_registro",
            "status": "precisa_revisao",
            "bbox_json": json.dumps(bbox_registro(0, 2)),
        }],
    )

    invalidas = repo.auditar_associacoes_qwen()
    deteccao = db.fetchone("SELECT * FROM ocr_deteccao WHERE execucao_id=?", (execucao,))
    historico = db.fetchone("SELECT * FROM ocr_execucao WHERE id=?", (execucao,))

    assert len(invalidas) == 1
    assert deteccao["ativo"] == 0
    assert deteccao["status"] == "descartado"
    assert deteccao["valor_tratado"] == "Begoíng dos Santos Teho"
    assert historico["ativo"] == 0
    assert "Begoíng" in historico["texto_bruto"]
    assert repo.tem_revisao_pendente(imagem_id, "qwen_associacao_invalida")
    db.close()


def test_sugestao_fica_neutra_e_confirmacao_fica_verde(tmp_path):
    db, repo, _, imagem_id, registros = _acervo_com_imagem(tmp_path)
    repo.salvar_metadado_tratado(
        imagem_id=imagem_id,
        registro_id=registros[1]["id"],
        tipo="nome_registrado",
        valor="Nome ainda duvidoso",
        confianca=0.45,
        fonte="qwen_nome_correcao",
        motor="qwen",
        status="precisa_revisao",
    )
    sugestao = repo.buscar_registros(termo=6802)[0]
    visual = estado_visual_registro(sugestao, False)
    assert sugestao["nome_confirmado"] == ""
    assert sugestao["nome_sugerido"] == "Nome ainda duvidoso"
    assert visual["fundo"] == "#ffffff"
    assert "NÃO CONFIRMADA" in visual["texto"]

    repo.salvar_metadado_tratado(
        imagem_id=imagem_id,
        registro_id=registros[1]["id"],
        tipo="nome_registrado",
        valor="Nome Corrigido",
        confianca=1.0,
        fonte="operador",
        motor="operador",
        status="confirmado",
    )
    confirmado = repo.buscar_registros(termo=6802)[0]
    visual = estado_visual_registro(confirmado, False)
    assert confirmado["nome_confirmado"] == "Nome Corrigido"
    assert visual["fundo"] == "#e8f5e9"
    db.close()


def test_triagem_rejeita_ruido_e_rebaixa_sugestao_antiga(tmp_path):
    assert not _nome_parece_legivel("OTe a7 Srwcoden An! Nz")
    assert not _nome_parece_legivel("Não consigo ler o nome")
    assert not _nome_parece_legivel("Aracaju - Sergipe")
    # Duas palavras formadas só por letras podem parecer plausíveis; por isso
    # o filtro lexical nunca é usado sozinho para promover uma sugestão.
    assert _nome_parece_legivel("Dlacaee pea")
    assert _nome_parece_legivel("Ana Beatriz de Souza")

    db, repo, livro_id, imagem_id, registros = _acervo_com_imagem(tmp_path)
    execucao = repo.criar_execucao_ocr(
        imagem_id=imagem_id,
        registro_id=registros[0]["id"],
        motor="ocr-nomes-rapido-v2",
        texto_bruto="que recebeu o nome de Dlacaee pea",
    )
    repo.salvar_deteccoes_ocr(
        execucao_id=execucao,
        imagem_id=imagem_id,
        registro_id=registros[0]["id"],
        deteccoes=[{
            "tipo": "nome_registrado",
            "valor_original": "Dlacaee pea",
            "valor_tratado": "Dlacaee pea",
            "valor_normalizado": "DLACAE PEA",
            "confianca": 0.72,
            "motor": "tesseract",
            "fonte": "ocr_nome_rapido",
            "status": "sugestao",
            "contexto": "Faixa do nome; concordância=1",
        }],
    )
    lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)
    quantidade = repo.rebaixar_sugestoes_rapidas_antigas(livro_id=livro_id)
    deteccao = db.fetchone(
        "SELECT ativo,status FROM ocr_deteccao WHERE execucao_id=?", (execucao,)
    )
    item = db.fetchone(
        "SELECT status FROM processamento_item WHERE lote_id=? AND registro_id=? AND etapa='ocr_nome_rapido'",
        (lote["id"], registros[0]["id"]),
    )
    assert quantidade == 1
    assert deteccao == {"ativo": 0, "status": "descartado"}
    assert item["status"] == "revisar"
    db.close()


def test_lote_materializa_1194_itens_e_retomada_nao_duplica(tmp_path):
    db = Database(tmp_path / "fila.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6, tipo_id=1, codigo="A-07", nome_capa="Nascimentos",
        registros_por_face=2, termo_inicial=6801, termo_final=7994,
    )
    agora = "2026-01-01T00:00:00"
    imagens = []
    registros = []
    termo = 6801
    for indice in range(597):
        imagem_id = indice + 1
        imagens.append((
            imagem_id, livro_id, imagem_id, f"sha-{imagem_id}", "registro",
            "unico", termo, termo + 1, agora,
        ))
        registros.extend((
            (imagem_id * 2 - 1, livro_id, imagem_id, 0, termo, "frente", agora, agora),
            (imagem_id * 2, livro_id, imagem_id, 1, termo + 1, "frente", agora, agora),
        ))
        termo += 2
    db.executemany(
        """
        INSERT INTO imagem
        (id, livro_id, ordem_captura, sha256, tipo_documento,
         duplicidade_status, termo_inicial, termo_final, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        imagens,
    )
    db.executemany(
        """
        INSERT INTO registro
        (id, livro_id, imagem_id, indice_na_imagem, termo, face,
         status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'inferido', ?, ?)
        """,
        registros,
    )

    lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)
    itens = repo.listar_itens_processamento(lote["id"], etapa="ocr_nome_rapido")
    assert len(itens) == 1194
    repo.atualizar_item_processamento(itens[0]["id"], status="processando")
    repo.preparar_retomada_lote(lote["id"])
    assert db.fetchone(
        "SELECT status FROM processamento_item WHERE id=?", (itens[0]["id"],)
    )["status"] == "pendente"

    mesmo_lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)
    assert mesmo_lote["id"] == lote["id"]
    assert db.fetchone(
        "SELECT COUNT(*) AS n FROM processamento_item WHERE lote_id=?",
        (lote["id"],),
    )["n"] == 1194
    db.close()


def test_trabalhador_persistente_processa_e_deixa_sugestao_sem_confirmar(
    tmp_path, monkeypatch
):
    db, repo, livro_id, _, registros = _acervo_com_imagem(tmp_path)
    lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)

    class ProvedorFalso:
        def __init__(self, name):
            self.name = name

        def recognize(self, _imagem, fast=True):
            assert fast
            return OCRResult(
                motor=self.name,
                texto_bruto="que recebeu o nome de Ana Beatriz de Souza",
                tempo_ms=3.0,
            )

    monkeypatch.setattr(
        "src.services.name_processing._provedores",
        lambda _settings: [ProvedorFalso("ocr-falso-a"), ProvedorFalso("ocr-falso-b")],
    )
    monkeypatch.setattr(
        "src.services.name_processing.retificar_formulario",
        lambda image: SimpleNamespace(
            image=image,
            confidence=1.0,
            left_line=None,
            right_line=None,
            reason="geometria simulada",
        ),
    )
    monkeypatch.setattr(
        "src.services.name_processing.modelo_qwen_instalado", lambda _path=None: False
    )

    settings = Settings(tmp_path / "config.yaml")
    settings.set("ocr", "name_qwen_threshold", 0.95)
    settings.set("ocr", "name_direct_qwen_books", [])
    resumo = NameBatchRunner(
        db_path=db.path,
        settings=settings,
        lote_id=int(lote["id"]),
        max_workers=2,
    ).run()
    itens_rapidos = repo.listar_itens_processamento(
        int(lote["id"]), etapa="ocr_nome_rapido"
    )
    itens_qwen = repo.listar_itens_processamento(int(lote["id"]), etapa="qwen_nome")

    assert resumo["status"] == "pausado"
    assert {item["status"] for item in itens_rapidos} == {"sugestao"}
    assert all(item["tentativas"] == 1 for item in itens_rapidos)
    assert len(itens_qwen) == 2
    assert {item["status"] for item in itens_qwen} == {"pendente"}
    for registro in registros:
        encontrado = repo.buscar_registros(termo=registro["termo"])[0]
        assert encontrado["nome_confirmado"] == ""
        assert encontrado["nome_sugerido"] == "Ana Beatriz de Souza"
    db.close()


def test_qwen_de_fila_le_apenas_nome_e_deixa_data_para_fila_posterior(tmp_path, monkeypatch):
    db, repo, livro_id, _, registros = _acervo_com_imagem(tmp_path)
    lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)

    class ProvedorFalso:
        def __init__(self, name):
            self.name = name

        def recognize(self, _imagem, fast=True):
            return OCRResult(
                motor=self.name,
                texto_bruto="que recebeu o nome de Ana Beatriz de Souza",
                tempo_ms=3.0,
            )

    class QwenFalso:
        def __init__(self, **_kwargs):
            pass

        def analisar_nome(self, _imagem):
            resultado = OCRResult(
                motor="qwen-falso",
                texto_bruto="Ana Beatriz de Souza",
                tempo_ms=120.0,
            )
            return "Ana Beatriz de Souza", resultado

        def liberar(self):
            pass

    monkeypatch.setattr(
        "src.services.name_processing._provedores",
        lambda _settings: [ProvedorFalso("ocr-falso-a"), ProvedorFalso("ocr-falso-b")],
    )
    monkeypatch.setattr(
        "src.services.name_processing.retificar_formulario",
        lambda image: SimpleNamespace(
            image=image,
            confidence=1.0,
            left_line=None,
            right_line=None,
            reason="geometria simulada",
        ),
    )
    monkeypatch.setattr("src.services.name_processing.modelo_qwen_instalado", lambda _path=None: True)
    monkeypatch.setattr("src.services.name_processing.QwenRecordAnalyzer", QwenFalso)
    monkeypatch.setattr(
        "src.services.name_processing._localizar_linhas_nome_pagina",
        lambda _image, _providers, total: {
            0: (bbox_linha_nome(0, total), "rotulo-simulado"),
            1: (bbox_linha_nome(1, total), "rotulo-simulado"),
        },
    )

    resumo = NameBatchRunner(
        db_path=db.path,
        settings=Settings(tmp_path / "config.yaml"),
        lote_id=int(lote["id"]),
        max_workers=2,
    ).run()

    assert resumo["status"] == "concluido"
    itens_rapidos = repo.listar_itens_processamento(
        int(lote["id"]), etapa="ocr_nome_rapido"
    )
    assert {item["status"] for item in itens_rapidos} == {"encaminhado_qwen"}
    for registro in registros:
        metadados = repo.listar_metadados_registro(registro["id"])
        nomes = [m for m in metadados if m["tipo"] == "nome_registrado"]
        datas = [m for m in metadados if m["tipo"] == "data"]
        assert nomes and nomes[0]["valor_tratado"] == "Ana Beatriz de Souza"
        assert not datas
    db.close()


def test_trabalhador_tenta_duas_vezes_e_reprocessamento_manual_reabre_falha(
    tmp_path, monkeypatch
):
    db, repo, livro_id, _, _ = _acervo_com_imagem(tmp_path)
    lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)

    class ProvedorComFalha:
        name = "ocr-falso"

        def recognize(self, _imagem, fast=True):
            raise RuntimeError("falha de leitura simulada")

    monkeypatch.setattr(
        "src.services.name_processing._provedores", lambda _settings: [ProvedorComFalha()]
    )
    monkeypatch.setattr(
        "src.services.name_processing.retificar_formulario",
        lambda image: SimpleNamespace(
            image=image,
            confidence=1.0,
            left_line=None,
            right_line=None,
            reason="geometria simulada",
        ),
    )
    monkeypatch.setattr(
        "src.services.name_processing.modelo_qwen_instalado", lambda _path=None: False
    )

    settings = Settings(tmp_path / "config.yaml")
    settings.set("ocr", "name_direct_qwen_books", [])
    NameBatchRunner(
        db_path=db.path,
        settings=settings,
        lote_id=int(lote["id"]),
        max_workers=2,
    ).run()
    falhas = repo.listar_itens_processamento(
        int(lote["id"]), etapa="ocr_nome_rapido", statuses=("falhou",)
    )
    assert len(falhas) == 2
    assert all(item["tentativas"] == 2 for item in falhas)
    assert all("falha de leitura simulada" in item["erro"] for item in falhas)

    assert repo.reprocessar_falhas_lote(int(lote["id"])) == 2
    reabertos = repo.listar_itens_processamento(
        int(lote["id"]), etapa="ocr_nome_rapido", statuses=("pendente",)
    )
    assert len(reabertos) == 2
    assert all(item["tentativas"] == 0 and not item["erro"] for item in reabertos)
    db.close()


def test_retificacao_alinha_copia_sem_alterar_imagem():
    base = np.full((1200, 900, 3), 238, np.uint8)
    for y in range(80, 1150, 35):
        cv2.line(base, (120, y), (820, y), (55, 55, 55), 2)
    cv2.line(base, (165, 30), (165, 1170), (40, 40, 40), 3)
    cv2.line(base, (675, 30), (675, 1170), (40, 40, 40), 3)
    matriz = cv2.getRotationMatrix2D((450, 600), -1.6, 1.0)
    inclinada = cv2.warpAffine(base, matriz, (900, 1200), borderValue=(238, 238, 238))
    antes = hashlib.sha256(inclinada.tobytes()).hexdigest()

    resultado = retificar_formulario(inclinada)

    assert resultado.applied
    assert resultado.confidence >= 0.70
    assert abs(resultado.angle_degrees) >= 0.5
    assert hashlib.sha256(inclinada.tobytes()).hexdigest() == antes


@pytest.mark.skipif(
    not Path(r"D:\A - 07\FRENTE\IMG_2025_07_02_14_14_07S.jpg").is_file(),
    reason="fotografia real do A-07 não está disponível",
)
def test_foto_real_6801_6802_tem_geometria_confiavel_e_original_intacto():
    caminho = Path(r"D:\A - 07\FRENTE\IMG_2025_07_02_14_14_07S.jpg")
    antes = hashlib.sha256(caminho.read_bytes()).hexdigest()
    imagem = cv2.imread(str(caminho))
    resultado = retificar_formulario(imagem)
    depois = hashlib.sha256(caminho.read_bytes()).hexdigest()

    assert resultado.applied
    assert resultado.confidence >= 0.70
    assert resultado.left_line and abs(resultado.left_line[0] - resultado.left_line[1]) < 0.02
    assert resultado.right_line and abs(resultado.right_line[0] - resultado.right_line[1]) < 0.02
    assert antes == depois


@pytest.mark.skipif(
    not Path(r"D:\A - 07\VERSO\IMG_2025_07_02_14_53_51S.jpg").is_file(),
    reason="fotografia real do A-07 não está disponível",
)
def test_localizador_real_acha_os_dois_nomes_sem_cair_em_aracaju():
    caminho = Path(r"D:\A - 07\VERSO\IMG_2025_07_02_14_53_51S.jpg")
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    geometria = retificar_formulario(imagem)

    from src.ocr.engines import RapidOCRProvider
    rapid = RapidOCRProvider()
    if not rapid.is_available():
        pytest.skip("RapidOCR indisponível")
    linhas = _localizar_linhas_nome_pagina(geometria.image, [rapid], total=2)

    assert set(linhas) == {0, 1}
    superior, inferior = linhas[0][0], linhas[1][0]
    assert 0.19 < superior[1] < 0.24
    assert 0.65 < inferior[1] < 0.71
    assert superior[2] <= 0.74 and inferior[2] <= 0.74


@pytest.mark.skipif(
    not Path(r"D:\A - 07\FRENTE\IMG_2025_07_02_14_14_07S.jpg").is_file(),
    reason="fotografia real do A-07 não está disponível",
)
def test_localizador_tolera_rotulo_recebeu_com_nome_mal_lido():
    caminho = Path(r"D:\A - 07\FRENTE\IMG_2025_07_02_14_14_07S.jpg")
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    geometria = retificar_formulario(imagem)

    from src.ocr.engines import RapidOCRProvider
    rapid = RapidOCRProvider()
    if not rapid.is_available():
        pytest.skip("RapidOCR indisponível")
    linhas = _localizar_linhas_nome_pagina(geometria.image, [rapid], total=2)

    assert set(linhas) == {0, 1}
    assert linhas[0][0][1] < 0.20
    assert 0.63 < linhas[1][0][1] < 0.69


def test_localizador_infere_a_outra_metade_quando_um_rotulo_falha():
    class RapidFalso:
        name = "rapidocr"

        def recognize(self, _image, fast=True):
            assert fast
            return SimpleNamespace(tokens=[SimpleNamespace(
                valor="que recebeu o nome de",
                confianca=0.91,
                bbox=[[0.18, 0.66], [0.52, 0.66], [0.52, 0.68], [0.18, 0.68]],
            )])

    linhas = _localizar_linhas_nome_pagina(
        np.full((1000, 700, 3), 240, np.uint8), [RapidFalso()], total=2
    )

    assert set(linhas) == {0, 1}
    assert abs((linhas[1][0][1] - linhas[0][0][1]) - 0.50) < 0.001
    assert linhas[0][0][0] == linhas[1][0][0] == 0.28


def _acervo_com_varias_imagens(tmp_path, n_imagens=4):
    """Livro com N imagens reais de 2 assentos cada (sem usar o banco de produção)."""
    db = Database(tmp_path / "background.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6, tipo_id=1, codigo="A-07", nome_capa="Nascimentos",
        registros_por_face=2, termo_inicial=6801,
        termo_final=6801 + 2 * n_imagens - 1,
    )
    imagens = []
    registros = []
    for indice in range(n_imagens):
        path = tmp_path / f"pagina_{indice}.jpg"
        imagem = np.full((600, 400, 3), 235, np.uint8)
        for y in range(40, 580, 25):
            cv2.line(imagem, (20, y), (380, y), (50, 50, 50), 1)
        cv2.imwrite(str(path), imagem)
        termo_a = 6801 + 2 * indice
        imagem_id = repo.registrar_imagem(
            livro_id=livro_id, ordem_captura=indice + 1,
            caminho_original=str(path), caminho_thumb=str(path),
            folha_estimada=indice + 1, face="frente",
            termo_inicial=termo_a, termo_final=termo_a + 1,
            duplicidade_status="unico",
        )
        imagens.append(imagem_id)
        registros.extend(repo.sincronizar_registros_imagem(imagem_id))
    return db, repo, livro_id, imagens, registros


def _mockar_motores(monkeypatch):
    class ProvedorFalso:
        def __init__(self, name):
            self.name = name

        def recognize(self, _imagem, fast=True):
            return OCRResult(
                motor=self.name,
                texto_bruto="que recebeu o nome de Ana Beatriz de Souza",
                tempo_ms=2.0,
            )

    class QwenFalso:
        def __init__(self, **_kwargs):
            pass

        def analisar_nome(self, _imagem):
            return "Ana Beatriz de Souza", OCRResult(
                motor="qwen-falso", texto_bruto="Ana Beatriz de Souza", tempo_ms=5.0
            )

        def liberar(self):
            pass

    monkeypatch.setattr(
        "src.services.name_processing._provedores",
        lambda _settings: [ProvedorFalso("ocr-falso-a"), ProvedorFalso("ocr-falso-b")],
    )
    monkeypatch.setattr(
        "src.services.name_processing.retificar_formulario",
        lambda image: SimpleNamespace(
            image=image,
            confidence=1.0,
            left_line=None,
            right_line=None,
            reason="geometria simulada",
        ),
    )
    monkeypatch.setattr(
        "src.services.name_processing._localizar_linhas_nome_pagina",
        lambda _image, _providers, total: {
            indice: (bbox_linha_nome(indice, total), "rotulo-simulado")
            for indice in range(total)
        },
    )
    monkeypatch.setattr(
        "src.services.name_processing.modelo_qwen_instalado", lambda _path=None: True
    )
    monkeypatch.setattr("src.services.name_processing.QwenRecordAnalyzer", QwenFalso)


def test_livro_processa_em_background_com_interrupcao_e_retomada_ate_concluir(
    tmp_path, monkeypatch
):
    """Cenario real: o processo roda em background, e interrompido no meio
    (simulando fechar o app ou queda), e um novo processo retoma o lote e
    conclui sem reprocessar nada."""
    db, repo, livro_id, _, registros = _acervo_com_varias_imagens(tmp_path, n_imagens=4)
    lote = repo.criar_ou_sincronizar_lote_nomes(livro_id)
    _mockar_motores(monkeypatch)

    settings = Settings(tmp_path / "config.yaml")
    settings.set("ocr", "name_qwen_threshold", 0.95)
    settings.set("ocr", "name_direct_qwen_books", [])

    processados = {"n": 0}

    def parar_apos_metade():
        return processados["n"] >= 4

    def on_progress(_resumo, _rotulo):
        processados["n"] += 1

    # Fase 1: primeiro "processo" para de forma segura depois da metade.
    resumo1 = NameBatchRunner(
        db_path=db.path,
        settings=settings,
        lote_id=int(lote["id"]),
        max_workers=2,
        should_stop=parar_apos_metade,
        on_progress=on_progress,
    ).run()
    assert resumo1["status"] == "pausado"
    rapido1 = repo.listar_itens_processamento(int(lote["id"]), etapa="ocr_nome_rapido")
    concluidos1 = [i for i in rapido1 if i["status"] == "sugestao"]
    pendentes1 = [i for i in rapido1 if i["status"] == "pendente"]
    presos1 = [i for i in rapido1 if i["status"] == "processando"]
    assert 0 < len(concluidos1) < 8
    assert len(pendentes1) == 8 - len(concluidos1)
    assert not presos1
    assert all(i["tentativas"] == 1 for i in concluidos1)

    # Simula uma queda no meio de um item: o resíduo fica como "processando".
    residuo = pendentes1[0]
    repo.atualizar_item_processamento(int(residuo["id"]), status="processando")

    # Fase 2: novo processo (runner novo, banco persistido) retoma e conclui.
    resumo2 = NameBatchRunner(
        db_path=db.path,
        settings=settings,
        lote_id=int(lote["id"]),
        max_workers=2,
    ).run()
    assert resumo2["status"] == "concluido"

    rapido2 = repo.listar_itens_processamento(int(lote["id"]), etapa="ocr_nome_rapido")
    qwen2 = repo.listar_itens_processamento(int(lote["id"]), etapa="qwen_nome")
    assert all(i["status"] not in ("pendente", "processando") for i in rapido2)
    assert all(i["status"] not in ("pendente", "processando") for i in qwen2)
    assert all(i["status"] == "revisar" for i in qwen2)

    # Nenhum assento foi processado mais de uma vez por etapa.
    por_registro = {
        int(registro["id"]): {"rapido": 0, "qwen": 0} for registro in registros
    }
    for linha in db.fetchall(
        "SELECT registro_id, motor FROM ocr_execucao WHERE registro_id IS NOT NULL"
    ):
        if int(linha["registro_id"]) not in por_registro:
            continue
        motor = str(linha["motor"])
        if motor == "ocr-nomes-rapido-v2":
            por_registro[int(linha["registro_id"])]["rapido"] += 1
        elif motor == "qwen-nome-faixa-v3":
            por_registro[int(linha["registro_id"])]["qwen"] += 1
    assert all(
        contagens == {"rapido": 1, "qwen": 1}
        for contagens in por_registro.values()
    )

    for registro in registros:
        metadados = repo.listar_metadados_registro(registro["id"])
        nomes = [m for m in metadados if m["tipo"] == "nome_registrado"]
        assert nomes and nomes[0]["valor_tratado"] == "Ana Beatriz de Souza"
    db.close()


def test_worker_lock_bloqueia_segundo_processo_no_mesmo_banco(tmp_path):
    """O lock exclusivo impede que dois processos consumam o mesmo lote."""
    db = Database(tmp_path / "lock.db")
    db.connect()
    db.close()

    obtido = threading.Event()
    liberar = threading.Event()
    segundo_resultado: dict = {}

    def primeiro():
        with _exclusive_worker_lock(db.path):
            obtido.set()
            liberar.wait(15)

    def segundo():
        try:
            with _exclusive_worker_lock(db.path):
                segundo_resultado["nao_bloqueou"] = True
        except RuntimeError as exc:
            segundo_resultado["bloqueado"] = str(exc)

    t1 = threading.Thread(target=primeiro)
    t1.start()
    assert obtido.wait(15)
    t2 = threading.Thread(target=segundo)
    t2.start()
    t2.join(15)
    liberar.set()
    t1.join(15)

    assert "bloqueado" in segundo_resultado
    assert "nao_bloqueou" not in segundo_resultado
    assert not t2.is_alive()
