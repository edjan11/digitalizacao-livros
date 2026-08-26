from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QScrollArea,
    QSizePolicy, QMessageBox,
)
from PySide6.QtGui import QFont

from ..database.repository import Repository
from ..session.scan_session import ScanSession
from .theme import (
    SUPERFICIE, BORDA, SECUNDARIO_BG, SECUNDARIO_BORDA, VERDE_ESMERALDA,
    VERDE_ESMERALDA_HOVER, STATUS_ATENCAO, TEXTO_PRIMARIO, TEXTO_NEON,
)

logger = logging.getLogger(__name__)


class BookSelector(QWidget):
    oficio_selecionado = Signal(int, str)
    tipo_selecionado = Signal(int, str, bool)
    subtipo_selecionado = Signal(int, str)
    livro_selecionado = Signal(int)
    criar_livro_clicked = Signal()
    continuar_sessao = Signal()

    def __init__(self, repo: Repository, session: ScanSession, parent=None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.session = session
        self._current_stack: list[QWidget] = []
        self._init_ui()
        if session.tem_sessao_ativa():
            self._show_session_banner()
        else:
            self._show_oficios()

    def _init_ui(self) -> None:
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._stack = QVBoxLayout()
        self._main_layout.addLayout(self._stack)
        self._main_layout.addStretch()

    def _clear_stack(self) -> None:
        while self._stack.count():
            item = self._stack.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _show_session_banner(self) -> None:
        self._clear_stack()
        banner = QFrame()
        banner.setObjectName("panel")
        banner.setStyleSheet(
            f"QFrame#panel {{ background-color: {SUPERFICIE}; border: 1px solid {BORDA}; "
            "border-radius: 8px; padding: 12px; }}"
        )
        bl = QVBoxLayout(banner)
        title = QLabel("Sessao anterior encontrada")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        bl.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        resumo = QLabel(self.session.resumo)
        resumo.setFont(QFont("Segoe UI", 11))
        resumo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(resumo)
        folha = self.session.ultima_folha
        face = self.session.ultima_face or "frente"
        info = QLabel(f"Folha: {folha} - {face.capitalize()}" if folha else "Pronto para iniciar")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(info)
        btn_layout = QHBoxLayout()
        btn_continuar = QPushButton("CONTINUAR")
        btn_continuar.setMinimumHeight(50)
        btn_continuar.setStyleSheet(
            f"QPushButton {{ background-color: {VERDE_ESMERALDA}; color: {TEXTO_PRIMARIO}; "
            f"font-size: 16px; font-weight: bold; border-radius: 6px; border: none; }} "
            f"QPushButton:hover {{ background-color: {VERDE_ESMERALDA_HOVER}; }}"
        )
        btn_continuar.clicked.connect(self.continuar_sessao.emit)
        btn_layout.addWidget(btn_continuar)
        btn_novo = QPushButton("Escolher outro livro")
        btn_novo.setMinimumHeight(50)
        btn_novo.clicked.connect(self._show_oficios)
        btn_layout.addWidget(btn_novo)
        bl.addLayout(btn_layout)
        self._stack.addWidget(banner)

    def _show_oficios(self) -> None:
        self._clear_stack()
        title = QLabel("Selecione o Oficio")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(title)
        grid = QVBoxLayout()
        oficios = self.repo.listar_oficios()
        for oficio in oficios:
            btn = QPushButton(oficio["nome"])
            btn.setMinimumHeight(55)
            btn.setStyleSheet(
                f"QPushButton {{ font-size: 14px; font-weight: 600; border: 2px solid {SECUNDARIO_BORDA}; "
                f"border-radius: 8px; background-color: {SUPERFICIE}; color: {TEXTO_PRIMARIO}; }} "
                f"QPushButton:hover {{ background-color: {SECUNDARIO_BG}; border: 2px solid #5A6678; }}"
            )
            btn.clicked.connect(lambda checked=False, o=oficio: self._on_oficio(o))
            grid.addWidget(btn)
        self._stack.addLayout(grid)

    def _on_oficio(self, oficio: dict) -> None:
        self.session.selecionar_oficio(oficio["id"])
        self.oficio_selecionado.emit(oficio["id"], oficio["nome"])
        self._show_tipos()

    def _show_tipos(self) -> None:
        self._clear_stack()
        nome_oficio = ""
        if self.session.oficio_id:
            o = self.repo.get_oficio(self.session.oficio_id)
            nome_oficio = o["nome"] if o else ""
        title = QLabel(nome_oficio)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(title)
        subtitle = QLabel("Selecione o tipo de registro")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(subtitle)
        grid = QVBoxLayout()
        tipos = self.repo.listar_tipos()
        for tipo in tipos:
            btn = QPushButton(tipo["nome"])
            btn.setMinimumHeight(55)
            btn.setStyleSheet(
                f"QPushButton {{ font-size: 14px; font-weight: 600; border: 2px solid {VERDE_ESMERALDA}; "
                f"border-radius: 8px; background-color: {SUPERFICIE}; color: {TEXTO_PRIMARIO}; }} "
                f"QPushButton:hover {{ background-color: {SECUNDARIO_BG}; }}"
            )
            subtipos = self.repo.listar_tipos(tipo["id"])
            tem_subtipos = len(subtipos) > 0
            btn.clicked.connect(
                lambda checked=False, t=tipo, sub=tem_subtipos: self._on_tipo(t, sub)
            )
            grid.addWidget(btn)
        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self._show_oficios)
        grid.addWidget(btn_voltar)
        self._stack.addLayout(grid)

    def _on_tipo(self, tipo: dict, tem_subtipos: bool) -> None:
        self.session.selecionar_tipo(tipo["id"])
        self.tipo_selecionado.emit(tipo["id"], tipo["nome"], tem_subtipos)
        if tem_subtipos:
            self._show_subtipos(tipo)
        else:
            self._show_livros()

    def _show_subtipos(self, tipo_pai: dict) -> None:
        self._clear_stack()
        title = QLabel(f"{tipo_pai['nome']}")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(title)
        subtitle = QLabel("Selecione o tipo especifico")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(subtitle)
        grid = QVBoxLayout()
        subtipos = self.repo.listar_tipos(tipo_pai["id"])
        for st in subtipos:
            btn = QPushButton(st["nome"])
            btn.setMinimumHeight(55)
            btn.setStyleSheet(
                f"QPushButton {{ font-size: 14px; font-weight: 600; border: 2px solid {STATUS_ATENCAO}; "
                f"border-radius: 8px; background-color: {SUPERFICIE}; color: {TEXTO_PRIMARIO}; }} "
                f"QPushButton:hover {{ background-color: {SECUNDARIO_BG}; }}"
            )
            btn.clicked.connect(lambda checked=False, s=st: self._on_subtipo(s))
            grid.addWidget(btn)
        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self._show_tipos)
        grid.addWidget(btn_voltar)
        self._stack.addLayout(grid)

    def _on_subtipo(self, subtipo: dict) -> None:
        self.session.selecionar_subtipo(subtipo["id"])
        self.subtipo_selecionado.emit(subtipo["id"], subtipo["nome"])
        self._show_livros()

    def _show_livros(self) -> None:
        self._clear_stack()
        nome_oficio = ""
        nome_tipo = ""
        if self.session.oficio_id:
            o = self.repo.get_oficio(self.session.oficio_id)
            nome_oficio = o["nome"] if o else ""
        if self.session.tipo_id:
            t = self.repo.get_tipo(self.session.tipo_id)
            nome_tipo = t["nome"] if t else ""
        title = QLabel(f"{nome_oficio}\n{nome_tipo}")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(title)
        subtitle = QLabel("Livros cadastrados")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(subtitle)
        grid = QVBoxLayout()
        livros = self.repo.listar_livros_por_categoria(
            self.session.oficio_id, self.session.tipo_id, self.session.subtipo_id
        )
        for livro in livros:
            status_emoji = "OK" if livro["status"] == "concluido" else ">"
            conferido = "  [Conferido]" if livro.get("conferido_em") else ""
            btn = QPushButton(f"{status_emoji}  {livro['codigo'] or 'Sem codigo'} - {livro['nome_capa'] or 'Sem nome'}{conferido}")
            btn.setMinimumHeight(50)
            btn.setStyleSheet(
                f"QPushButton {{ font-size: 13px; border: 1px solid {BORDA}; border-radius: 6px; "
                f"background-color: {SUPERFICIE}; color: {TEXTO_PRIMARIO}; "
                "text-align: left; padding-left: 12px; } "
                f"QPushButton:hover {{ background-color: {SECUNDARIO_BG}; border: 1px solid #5A6678; }}"
            )
            btn.clicked.connect(lambda checked=False, l=livro: self._on_livro(l))
            grid.addWidget(btn)
        btn_novo = QPushButton("+ NOVO LIVRO")
        btn_novo.setMinimumHeight(55)
        btn_novo.setStyleSheet(
            f"QPushButton {{ font-size: 14px; border: 2px dashed {VERDE_ESMERALDA}; border-radius: 8px; "
            f"background-color: {SUPERFICIE}; color: {VERDE_ESMERALDA}; font-weight: bold; }} "
            f"QPushButton:hover {{ background-color: {SECUNDARIO_BG}; }}"
        )
        btn_novo.clicked.connect(self.criar_livro_clicked.emit)
        grid.addWidget(btn_novo)
        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self._show_tipos)
        grid.addWidget(btn_voltar)
        scroll = QScrollArea()
        container = QWidget()
        container.setLayout(grid)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        self._stack.addWidget(scroll)

    def _on_livro(self, livro: dict) -> None:
        self.session.selecionar_livro(livro["id"])
        self.livro_selecionado.emit(livro["id"])
