from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFrame

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.book_review_dialog import BookReviewDialog
from tests.test_consulta import _acervo_com_imagem


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _cartoes(dialog: BookReviewDialog) -> int:
    """QLabel tambem herda de QFrame; contamos apenas os cartoes nomeados."""
    return sum(
        1
        for i in range(dialog._container_layout.count())
        if dialog._container_layout.itemAt(i).widget()
        and dialog._container_layout.itemAt(i).widget().objectName() == "card_pendencia"
    )


def test_painel_mostra_resumo_e_cartoes_por_tipo(tmp_path):
    _app()
    db, repo, livro_id, imagem_id, _registros = _acervo_com_imagem(tmp_path)
    repo.criar_revisao(imagem_id=imagem_id, tipo="refazer_captura",
                       detalhes="reflexo forte sobre a pagina")
    repo.criar_revisao(imagem_id=imagem_id, tipo="termo_incerto",
                       detalhes="Termo OCR: 6801 | Esperado: 6800")

    dlg = BookReviewDialog(repo, None, livro_id)

    resumo = dlg.lbl_resumo.text()
    assert "aprovadas" in resumo and "revisar" in resumo
    assert "recapturar" in resumo and "faltantes" in resumo
    assert "1 de 600 faces capturadas" in dlg.progresso.format()
    assert _cartoes(dlg) == 2
    assert "pendencia(s)" in dlg.lbl_rodape.text()
    assert not dlg.btn_concluir.isEnabled()

    # Filtro por tipo mostra apenas o cartao correspondente.
    dlg._aplicar_filtro("refazer_captura")
    assert _cartoes(dlg) == 1
    dlg._aplicar_filtro("termo_incerto")
    assert _cartoes(dlg) == 1
    dlg._aplicar_filtro("todos")
    assert _cartoes(dlg) == 2
    dlg.close()
    db.close()


def test_painel_livro_pronto_quando_sem_pendencias(tmp_path):
    _app()
    db, repo, livro_id, _imagem_id, _registros = _acervo_com_imagem(tmp_path)

    dlg = BookReviewDialog(repo, None, livro_id)
    assert "Livro pronto" in dlg.lbl_rodape.text()
    assert dlg.btn_concluir.isEnabled()
    assert "0" in dlg.lbl_resumo.text()
    dlg.close()
    db.close()


def test_resolver_pendencia_atualiza_painel_para_pronto(tmp_path):
    _app()
    db, repo, livro_id, imagem_id, _registros = _acervo_com_imagem(tmp_path)
    revisao_id = repo.criar_revisao(imagem_id=imagem_id, tipo="qualidade",
                                    detalhes="exposicao inadequada")

    dlg = BookReviewDialog(repo, None, livro_id)
    assert not dlg.btn_concluir.isEnabled()

    dlg._resolver(revisao_id)
    assert "Livro pronto" in dlg.lbl_rodape.text()
    assert dlg.btn_concluir.isEnabled()
    assert _cartoes(dlg) == 0
    dlg.close()
    db.close()


def test_concluir_marca_livro_conferido(monkeypatch, tmp_path):
    _app()
    db, repo, livro_id, _imagem_id, _registros = _acervo_com_imagem(tmp_path)
    monkeypatch.setattr(
        "src.ui.book_review_dialog.QMessageBox.question",
        lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.StandardButton.Yes,
    )

    dlg = BookReviewDialog(repo, None, livro_id)
    assert not repo.livro_conferido(livro_id)
    dlg._concluir()
    assert repo.livro_conferido(livro_id)
    assert dlg.result() == 1  # accept() foi chamado
    db.close()


def test_listar_revisoes_pendentes_filtra_por_livro(tmp_path):
    _app()
    db, repo, livro_a, _im, _reg = _acervo_com_imagem(tmp_path)
    db2, repo2, livro_b, imagem_b, _reg2 = _acervo_com_imagem(tmp_path)
    repo.criar_revisao(imagem_id=imagem_b, tipo="refazer_captura", detalhes="x")

    assert len(repo.listar_revisoes_pendentes(livro_a)) == 0
    assert len(repo.listar_revisoes_pendentes(livro_b)) == 1
    assert len(repo.listar_revisoes_pendentes()) == 1
    db.close()
    db2.close()
