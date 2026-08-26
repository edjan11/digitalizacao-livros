from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QSpinBox,
    QComboBox, QCheckBox, QTextEdit, QPushButton, QLabel, QMessageBox,
)
from PySide6.QtGui import QFont

from ..database.repository import Repository
from ..session.scan_session import ScanSession
from .theme import VERDE_ESMERALDA, VERDE_ESMERALDA_HOVER, TEXTO_PRIMARIO, TEXTO_NEON

logger = logging.getLogger(__name__)


class BookDialog(QDialog):
    def __init__(self, repo: Repository, session: ScanSession, parent=None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.session = session
        self.setWindowTitle("Cadastrar Novo Livro")
        self.setMinimumSize(500, 600)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Novo Livro")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        form = QFormLayout()
        self.codigo = QLineEdit()
        self.codigo.setPlaceholderText("Ex: A-12")
        form.addRow("Codigo/Referencia:", self.codigo)

        self.nome_capa = QLineEdit()
        self.nome_capa.setPlaceholderText("Ex: Nascimentos 1920-1925")
        form.addRow("Nome na capa:", self.nome_capa)

        self.total_folhas = QSpinBox()
        self.total_folhas.setRange(1, 9999)
        self.total_folhas.setValue(300)
        form.addRow("Total de folhas:", self.total_folhas)

        self.primeira_folha = QSpinBox()
        self.primeira_folha.setRange(1, 9999)
        self.primeira_folha.setValue(1)
        form.addRow("Primeira folha:", self.primeira_folha)

        self.ultima_folha = QSpinBox()
        self.ultima_folha.setRange(1, 9999)
        self.ultima_folha.setValue(300)
        form.addRow("Ultima folha:", self.ultima_folha)

        self.frente_verso = QCheckBox("Possui frente e verso")
        self.frente_verso.setChecked(True)
        form.addRow("Frente/Verso:", self.frente_verso)

        self.registros_por_face = QSpinBox()
        self.registros_por_face.setRange(1, 10)
        self.registros_por_face.setValue(2)
        form.addRow("Registros por face:", self.registros_por_face)

        self.termo_inicial = QSpinBox()
        self.termo_inicial.setRange(1, 9999999)
        self.termo_inicial.setValue(1)
        form.addRow("Termo inicial:", self.termo_inicial)

        self.termo_final = QSpinBox()
        self.termo_final.setRange(1, 9999999)
        self.termo_final.setValue(1200)
        form.addRow("Termo final:", self.termo_final)

        self.observacoes = QTextEdit()
        self.observacoes.setMaximumHeight(80)
        self.observacoes.setPlaceholderText("Observacoes e excecoes conhecidas...")
        form.addRow("Observacoes:", self.observacoes)

        layout.addLayout(form)

        total_registros = (
            self.total_folhas.value()
            * (2 if self.frente_verso.isChecked() else 1)
            * self.registros_por_face.value()
        )
        self.lbl_estimativa = QLabel(f"Estimativa: ~{total_registros} registros")
        self.lbl_estimativa.setStyleSheet(f"color: {TEXTO_NEON}; font-weight: bold;")
        self.lbl_estimativa.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_estimativa)

        self.total_folhas.valueChanged.connect(self._recalcular)
        self.frente_verso.toggled.connect(self._recalcular)
        self.registros_por_face.valueChanged.connect(self._recalcular)

        btn_layout = QHBoxLayout()
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancelar)

        btn_salvar = QPushButton("Salvar e Iniciar")
        btn_salvar.setStyleSheet(
            f"QPushButton {{ background-color: {VERDE_ESMERALDA}; color: {TEXTO_PRIMARIO}; "
            f"font-weight: bold; padding: 8px 24px; border-radius: 6px; border: none; }} "
            f"QPushButton:hover {{ background-color: {VERDE_ESMERALDA_HOVER}; }}"
        )
        btn_salvar.clicked.connect(self._salvar)
        btn_layout.addWidget(btn_salvar)
        layout.addLayout(btn_layout)

    def _recalcular(self) -> None:
        total = (
            self.total_folhas.value()
            * (2 if self.frente_verso.isChecked() else 1)
            * self.registros_por_face.value()
        )
        self.lbl_estimativa.setText(f"Estimativa: ~{total} registros")

    def _salvar(self) -> None:
        codigo = self.codigo.text().strip()
        if not codigo:
            QMessageBox.warning(self, "Erro", "Informe o codigo do livro.")
            return
        livro_id = self.repo.criar_livro(
            oficio_id=self.session.oficio_id,
            tipo_id=self.session.tipo_id,
            subtipo_id=self.session.subtipo_id,
            codigo=codigo,
            nome_capa=self.nome_capa.text().strip(),
            total_folhas=self.total_folhas.value(),
            primeira_folha=self.primeira_folha.value(),
            ultima_folha=self.ultima_folha.value(),
            frente_verso=1 if self.frente_verso.isChecked() else 0,
            registros_por_face=self.registros_por_face.value(),
            termo_inicial=self.termo_inicial.value(),
            termo_final=self.termo_final.value(),
            observacoes=self.observacoes.toPlainText().strip(),
        )
        self.session.selecionar_livro(livro_id)
        self.accept()
