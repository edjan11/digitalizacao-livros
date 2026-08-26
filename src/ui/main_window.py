from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QPushButton, QMessageBox, QStatusBar,
)

from ..config.settings import Settings
from ..database.connection import Database
from ..database.repository import Repository
from ..session.scan_session import ScanSession
from ..services.scan_pipeline import ScanPipeline
from ..watcher.folder_watcher import FolderWatcher
from watcher_component import carregar_pasta
from .book_selector import BookSelector
from .book_dialog import BookDialog
from .scan_screen import ScanScreen
from .review_dialog import ReviewDialog
from .lab_dialog import LabDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Digitalizador de Livros v2.0 Enterprise")
        self.setMinimumSize(1200, 760)
        self.resize(1440, 900)

        self._db: Database | None = None
        self._repo: Repository | None = None
        self._session: ScanSession | None = None
        self._pipeline: ScanPipeline | None = None
        self._watcher: FolderWatcher | None = None

        self._init_db()
        self._init_ui()
        self._init_watcher()

    def _init_db(self) -> None:
        self._db = Database()
        self._db.connect()
        self._repo = Repository(self._db)
        self._session = ScanSession(self._repo)

    def _init_ui(self) -> None:
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._selector = BookSelector(self._repo, self._session)
        self._selector.oficio_selecionado.connect(self._on_oficio)
        self._selector.tipo_selecionado.connect(self._on_tipo)
        self._selector.subtipo_selecionado.connect(self._on_subtipo)
        self._selector.criar_livro_clicked.connect(self._on_criar_livro)
        self._selector.livro_selecionado.connect(self._on_livro_selecionado)
        self._selector.continuar_sessao.connect(self._on_continuar_sessao)
        self._stack.addWidget(self._selector)

        self.statusBar().showMessage("Pronto")

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 4, 8, 4)
        titulo = QLabel("DIGITALIZADOR DE LIVROS")
        titulo.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        titulo.setStyleSheet("color: #F0F4F8;")
        top_bar.addWidget(titulo)
        top_bar.addStretch()
        btn_lab = QPushButton("Laboratorio")
        btn_lab.setToolTip("Ferramenta de teste de OCR/HTR")
        btn_lab.clicked.connect(self._abrir_lab)
        top_bar.addWidget(btn_lab)
        btn_config = QPushButton("Configuracoes")
        btn_config.setToolTip("Pastas de destino, scanner e OCR")
        btn_config.clicked.connect(self._abrir_config)
        top_bar.addWidget(btn_config)
        if self._selector.layout() is None:
            self._selector.setLayout(QVBoxLayout())
        self._selector.layout().insertLayout(0, top_bar)

    def _init_watcher(self) -> None:
        self._watcher = FolderWatcher()
        self._watcher.imagem_detectada.connect(self._on_nova_imagem)

    @Slot(int, str)
    def _on_oficio(self, oficio_id: int, nome: str) -> None:
        self.statusBar().showMessage(f"Oficio: {nome}")

    @Slot(int, str, bool)
    def _on_tipo(self, tipo_id: int, nome: str, tem_subtipos: bool) -> None:
        self.statusBar().showMessage(f"Tipo: {nome}")

    @Slot(int, str)
    def _on_subtipo(self, subtipo_id: int, nome: str) -> None:
        self.statusBar().showMessage(f"Subtipo: {nome}")

    @Slot()
    def _on_criar_livro(self) -> None:
        dlg = BookDialog(self._repo, self._session, self)
        if dlg.exec():
            self._iniciar_digitalizacao()

    @Slot(int)
    def _on_livro_selecionado(self, livro_id: int) -> None:
        self._iniciar_digitalizacao()

    @Slot()
    def _on_continuar_sessao(self) -> None:
        self._iniciar_digitalizacao()

    def _iniciar_digitalizacao(self) -> None:
        livro = self._session.livro
        if not livro:
            return
        acervo_root = Path(self.settings.get("acervo", "root_path", r"D:\AcervoLivros"))
        try:
            acervo_root.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            acervo_root = Path.home() / "DigitalizadorLivros" / "Acervo"
            acervo_root.mkdir(parents=True, exist_ok=True)
        self._pipeline = ScanPipeline(self._repo, self._session, acervo_root, self.settings)

        czur_path = carregar_pasta(self.settings, "czur", "watch_folder")
        if not str(czur_path):
            czur_path = Path.home() / "DigitalizadorLivros" / "CZUR" / "Scans"
        czur_path = self._garantir_pasta(czur_path, "pasta monitorada (CZUR)")
        self._watcher.iniciar(str(czur_path))

        self._scan_screen = ScanScreen(self._repo, self._session, self._pipeline)
        self._scan_screen.set_scanner_status(True)
        self._scan_screen.voltar_clicked.connect(self._voltar_selecao)
        self._scan_screen.revisao_clicked.connect(self._abrir_revisao)
        self._stack.addWidget(self._scan_screen)
        self._stack.setCurrentWidget(self._scan_screen)
        self.statusBar().showMessage("Digitalização automática de mesa ativada.")

    def _garantir_pasta(self, pasta: Path, descricao: str) -> Path:
        """Garante uma pasta gravavel, recaindo para a home do usuario.

        O caminho configurado pode apontar para uma midia protegida (ex.:
        leitora de CD/DVD em D:\\) e falhar com PermissionError. Em vez de
        travar silenciosamente a abertura do livro, usamos um subdiretorio da
        pasta do usuario e avisamos o operador.
        """
        pasta = Path(pasta)
        try:
            pasta.mkdir(parents=True, exist_ok=True)
            return pasta
        except (OSError, PermissionError) as exc:
            fallback = Path.home() / "DigitalizadorLivros" / pasta.name
            try:
                fallback.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                pass
            QMessageBox.warning(
                self,
                "Pasta indisponivel",
                f"Nao foi possivel usar a {descricao}:\n{pasta}\n({exc})\n\n"
                f"Usando instead: {fallback}",
            )
            return fallback    @Slot(str)
    def _on_nova_imagem(self, path: str) -> None:
        if self._scan_screen and self._stack.currentWidget() == self._scan_screen:
            self._scan_screen.processar_nova_imagem(path)
            self.statusBar().showMessage(f"Imagem detectada: {Path(path).name}")

    def _voltar_selecao(self) -> None:
        if self._watcher:
            self._watcher.parar()
        self._stack.setCurrentWidget(self._selector)
        self.statusBar().showMessage("Pronto")

    def _abrir_revisao(self) -> None:
        dlg = ReviewDialog(self._repo, self._pipeline, self)
        dlg.exec()

    def _abrir_lab(self) -> None:
        dlg = LabDialog(self)
        dlg.exec()

    def _abrir_config(self) -> None:
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(self.settings, self)
        dlg.exec()

    def closeEvent(self, event) -> None:
        if self._watcher:
            self._watcher.parar()
        if self._db:
            self._db.close()
        event.accept()
