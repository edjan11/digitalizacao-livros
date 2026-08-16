from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.ocr.got_ocr_engine import _texto_repetitivo
from src.ocr.qwen_vl_engine import _campos_json_qwen, _limpar_resposta, preparar_imagem_qwen
from src.ui.image_viewer import ImageViewer
from src.consulta.main_window import _tratar_termo_qwen


def test_descarta_loop_repetitivo_do_got():
    assert _texto_repetitivo("\n".join(["0219"] * 20))
    assert not _texto_repetitivo("linha um\nlinha dois\nlinha três")


def test_limpa_apenas_invólucro_da_resposta_qwen():
    assert _limpar_resposta('```text\nJoão da Silva\n```') == "João da Silva"
    assert _limpar_resposta('"Maria de Souza"') == "Maria de Souza"


def test_viewer_converte_selecao_para_coordenadas_relativas():
    app = QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    viewer.resize(500, 300)
    viewer.set_image_array(np.full((200, 400, 3), 255, np.uint8))
    viewer._display_scale = 0.5
    viewer._on_selection_finished(QRect(50, 25, 100, 50))

    bbox = viewer.selected_relative_rect()
    assert bbox is not None
    assert bbox == (0.25, 0.25, 0.75, 0.75)
    viewer.close()
    assert app is not None


def test_moldura_e_zoom_nao_alteram_imagem_original():
    app = QApplication.instance() or QApplication([])
    imagem = np.zeros((240, 480, 3), dtype=np.uint8)
    imagem[:, :, 1] = 127
    antes = hashlib.sha256(imagem.tobytes()).hexdigest()
    viewer = ImageViewer()
    viewer.resize(640, 400)
    viewer.set_image_array(imagem)
    viewer._set_selection_scene(QRectF(48, 24, 192, 96))
    bbox_antes = viewer.selected_relative_rect()
    viewer.set_zoom_percent(200)
    bbox_depois = viewer.selected_relative_rect()
    assert bbox_antes == bbox_depois == (0.1, 0.1, 0.5, 0.5)
    assert hashlib.sha256(imagem.tobytes()).hexdigest() == antes
    viewer.set_selection_mode(False)
    assert viewer.selection_mode() is False
    viewer.close()
    assert app is not None


def test_moldura_do_registro_exclui_coluna_de_averbacoes():
    app = QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    viewer.set_image_array(
        np.full((200, 400, 3), 255, np.uint8),
        destaque_indice=0,
        total_registros=2,
        destaque_x_rel=(0.055, 0.74),
    )
    rect = viewer._highlight_item.rect()
    assert round(rect.left() / 400, 3) == 0.055
    assert round(rect.right() / 400, 3) == 0.74
    viewer.close()
    assert app is not None


def test_moldura_padrao_equivale_ao_recorte_do_registro():
    from src.imaging.record_regions import bbox_registro

    app = QApplication.instance() or QApplication([])
    viewer = ImageViewer()
    viewer.set_image_array(
        np.full((200, 400, 3), 255, np.uint8),
        destaque_indice=1,
        total_registros=2,
    )
    rect = viewer._highlight_item.rect()
    bx1, by1, bx2, by2 = bbox_registro(1, 2)
    assert round(rect.left() / 400, 3) == round(bx1, 3)
    assert round(rect.right() / 400, 3) == round(bx2, 3)
    assert rect.top() == 200 * 0.490  # int(by1 * altura)
    assert rect.bottom() == 199.0
    viewer.close()
    assert app is not None


def test_preparo_qwen_e_temporario_e_preserva_cor_da_tinta():
    imagem = np.zeros((80, 120, 3), dtype=np.uint8)
    imagem[:, :] = (180, 180, 210)
    imagem[20:24, 10:100] = (180, 40, 20)  # tinta azul em BGR
    original = imagem.copy()
    preparada = preparar_imagem_qwen(imagem)
    assert preparada.shape == imagem.shape
    assert preparada.dtype == imagem.dtype
    assert np.array_equal(imagem, original)
    assert not np.array_equal(preparada, imagem)


def test_qwen_registro_parseia_somente_campos_basicos():
    campos = _campos_json_qwen(
        '```json\n{"nome_registrado":"Ana Cruz", "termo":"6801", "nome_mae":"Maria Cruz", "data_registro":"25 de maio de 1983"}\n```'
    )
    assert campos == {
        "nome_registrado": "Ana Cruz",
        "termo": "6801",
        "nome_mae": "Maria Cruz",
        "data_registro": "25 de maio de 1983",
    }


def test_qwen_registro_recupera_json_interrompido():
    campos = _campos_json_qwen(
        'json {"nome_registrado":"Jose Domingos Sertao", '
        '"termo":"6.801", "nome_mae":"Maria Flavia da Silva Cruz", '
        '"data_registro":"25 de maio de mil novecentos'
    )
    assert campos["nome_registrado"] == "Jose Domingos Sertao"
    assert campos["termo"] == "6.801"
    assert campos["nome_mae"] == "Maria Flavia da Silva Cruz"
    assert campos["data_registro"] == "25 de maio de mil novecentos"


def test_termo_qwen_corrige_ruido_com_a_sequencia_sem_inventar():
    assert _tratar_termo_qwen("6.8001", 6801)[0] == "6801"
    assert "ajustado" in _tratar_termo_qwen("6.8001", 6801)[1]
    assert _tratar_termo_qwen("9999", 6801)[0] == "9999"
