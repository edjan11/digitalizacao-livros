"""Painel de conferencia de um livro (M4).

Mostra o resumo do livro (aprovadas/revisar/recapturar/faltantes), a barra de
progresso, filtros por tipo de pendencia e cartoes com ABRIR FOTO / RECAPTURAR /
IGNORAR. Quando zera as pendencias, oferece "Fechar e marcar conferido".
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..database.repository import Repository
from ..services.scan_pipeline import ScanPipeline
from .image_viewer import ImageViewer
from .review_dialog import (
    TIPO_CORES,
    TIPO_NOMES,
    abrir_camera_refoto,
    aplicar_refoto,
    rotulo_revisao,
)

FILTROS = [
    ("todos", "Todos"),
    ("refazer_captura", "Refazer foto"),
    ("qualidade", "Qualidade"),
    ("termo_incerto", "Termo incerto"),
    ("folha_incerta", "Folha incerta"),
    ("duplicidade", "Duplicidade"),
    ("nome_incerto", "Nome incerto"),
    ("ocr_falha", "OCR"),
]


class _FotoDialog(QDialog):
    """Visualizador simples com zoom/pan para 'Abrir foto'."""

    def __init__(self, caminho: Path, titulo: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self.setMinimumSize(900, 620)
        layout = QVBoxLayout(self)
        self.viewer = ImageViewer()
        self.viewer.set_selection_mode(False)
        self.viewer.set_image_path(str(caminho))
        layout.addWidget(self.viewer, 1)
        botoes = QHBoxLayout()
        for texto, acao in (
            ("Ajustar", self.viewer.fit_to_window),
            ("100%", lambda: self.viewer.set_zoom_percent(100)),
            ("Fechar", self.accept),
        ):
            botao = QPushButton(texto)
            botao.clicked.connect(acao)
            botoes.addWidget(botao)
        botoes.addStretch()
        layout.addLayout(botoes)


class BookReviewDialog(QDialog):
    """Conferencia de um livro: resumo, filtros e acoes por pendencia."""

    def __init__(
        self,
        repo: Repository,
        pipeline: ScanPipeline | None,
        livro_id: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self.pipeline = pipeline
        self.livro_id = int(livro_id)
        self._filtro = "todos"
        self._revisoes: list[dict] = []
        self.setWindowTitle("Conferir livro")
        self.setMinimumSize(980, 680)
        self._init_ui()
        self._carregar()

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.lbl_titulo = QLabel()
        self.lbl_titulo.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(self.lbl_titulo)

        self.progresso = QProgressBar()
        self.progresso.setMaximum(100)
        layout.addWidget(self.progresso)

        self.lbl_resumo = QLabel()
        self.lbl_resumo.setWordWrap(True)
        self.lbl_resumo.setStyleSheet(
            "padding: 8px; background: #eef2f7; border-radius: 5px; font-size: 13px;"
        )
        layout.addWidget(self.lbl_resumo)

        filtros = QHBoxLayout()
        self.btn_filtros: dict[str, QPushButton] = {}
        for chave, rotulo in FILTROS:
            botao = QPushButton(rotulo)
            botao.setCheckable(True)
            botao.clicked.connect(lambda _c, k=chave: self._aplicar_filtro(k))
            filtros.addWidget(botao)
            self.btn_filtros[chave] = botao
        filtros.addStretch()
        layout.addLayout(filtros)

        self._scroll = QScrollArea()
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self.lbl_rodape = QLabel()
        self.lbl_rodape.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_rodape.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(self.lbl_rodape)

        botoes = QHBoxLayout()
        self.btn_concluir = QPushButton("Fechar e marcar conferido")
        self.btn_concluir.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-weight: bold; "
            "padding: 8px 18px; border-radius: 5px; }"
        )
        self.btn_concluir.clicked.connect(self._concluir)
        botoes.addWidget(self.btn_concluir)
        fechar = QPushButton("Fechar")
        fechar.clicked.connect(self.accept)
        botoes.addWidget(fechar)
        botoes.addStretch()
        layout.addLayout(botoes)

    # --------------------------------------------------------------- dados

    def _carregar(self) -> None:
        livro = self.repo.get_livro(self.livro_id) or {}
        self._revisoes = self.repo.listar_revisoes_pendentes(self.livro_id)
        resumo = self.repo.resumo_conferencia_livro(self.livro_id)

        codigo = livro.get("codigo") or "?"
        nome = livro.get("nome_capa") or ""
        conferido = " ✓" if resumo.get("conferido_em") else ""
        self.lbl_titulo.setText(f"Livro {codigo} — {nome}{conferido}")

        esperadas = max(1, int(resumo.get("esperadas_faces") or 1))
        capturadas = int(resumo.get("capturadas") or 0)
        self.progresso.setValue(min(100, int(capturadas * 100 / esperadas)))
        self.progresso.setFormat(
            f"{capturadas} de {esperadas} faces capturadas"
        )

        self.lbl_resumo.setText(
            f"<b>{resumo.get('aprovadas')}</b> aprovadas  ·  "
            f"<b style='color:#ef8c00'>{resumo.get('revisar')}</b> revisar  ·  "
            f"<b style='color:#c62828'>{resumo.get('recapturar')}</b> recapturar  ·  "
            f"<b style='color:#607d8b'>{resumo.get('faltantes')}</b> faltantes"
        )

        total = sum(
            1 for r in self._revisoes
            if self._filtro == "todos" or r.get("tipo") == self._filtro
        )
        pronto = not self._revisoes
        if pronto:
            self.lbl_rodape.setText("Livro pronto ✓ — nenhuma pendencia")
            self.lbl_rodape.setStyleSheet("color: #2e7d32;")
        else:
            self.lbl_rodape.setText(f"{total} pendencia(s)")
            self.lbl_rodape.setStyleSheet("color: #c62828;")
        self.btn_concluir.setEnabled(pronto)

        self._renderizar()

    def _aplicar_filtro(self, chave: str) -> None:
        self._filtro = chave
        for k, botao in self.btn_filtros.items():
            botao.setChecked(k == chave)
        self._renderizar()

    def _renderizar(self) -> None:
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        filtradas = [
            r for r in self._revisoes
            if self._filtro == "todos" or r.get("tipo") == self._filtro
        ]
        if not filtradas:
            vazio = QLabel(
                "Nenhuma pendencia neste filtro."
                if self._revisoes else "Nenhuma pendencia para este livro."
            )
            vazio.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vazio.setFont(QFont("Segoe UI", 12))
            self._container_layout.addWidget(vazio)
            self._container_layout.addStretch()
            return
        for revisao in filtradas:
            self._container_layout.addWidget(self._criar_card(revisao))
        self._container_layout.addStretch()

    # --------------------------------------------------------------- cartoes

    def _criar_card(self, revisao: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("card_pendencia")
        card.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 6px; "
            "background: white; margin: 4px; }"
        )
        layout = QVBoxLayout(card)

        tipo = revisao.get("tipo", "desconhecido")
        header = QLabel(rotulo_revisao(revisao))
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {TIPO_CORES.get(tipo, '#757575')};")
        layout.addWidget(header)

        corpo = QHBoxLayout()
        thumb = Path(revisao.get("caminho_thumb") or "")
        original = Path(revisao.get("caminho_original") or "")
        imagem_path = thumb if thumb.is_file() else original
        if imagem_path.is_file():
            pix = QPixmap(str(imagem_path))
            if not pix.isNull():
                miniatura = QLabel()
                miniatura.setAlignment(Qt.AlignmentFlag.AlignCenter)
                miniatura.setPixmap(
                    pix.scaled(
                        320, 150, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                corpo.addWidget(miniatura)

        detalhes = QLabel(revisao.get("detalhes", ""))
        detalhes.setWordWrap(True)
        corpo.addWidget(detalhes, 1)
        layout.addLayout(corpo)

        acoes = QHBoxLayout()
        btn_abrir = QPushButton("Abrir foto")
        btn_abrir.clicked.connect(lambda: self._abrir_foto(revisao))
        acoes.addWidget(btn_abrir)
        if tipo in ("refazer_captura", "qualidade") and self.pipeline is not None:
            btn_camera = QPushButton("Recapturar com camera")
            btn_camera.clicked.connect(lambda: self._recapturar_camera(revisao))
            acoes.addWidget(btn_camera)
            btn_arquivo = QPushButton("Recapturar com arquivo")
            btn_arquivo.clicked.connect(lambda: self._recapturar_arquivo(revisao))
            acoes.addWidget(btn_arquivo)
        btn_ignorar = QPushButton("Ignorar (nao e problema)")
        btn_ignorar.clicked.connect(lambda: self._resolver(revisao["id"]))
        acoes.addWidget(btn_ignorar)
        acoes.addStretch()
        layout.addLayout(acoes)

        return card

    # ---------------------------------------------------------------- acoes

    def _abrir_foto(self, revisao: dict) -> None:
        original = Path(revisao.get("caminho_original") or "")
        if original.is_file():
            _FotoDialog(original, rotulo_revisao(revisao), self).exec()

    def _recapturar_camera(self, revisao: dict) -> None:
        abrir_camera_refoto(self, self.pipeline, revisao, on_aplicado=self._carregar)

    def _recapturar_arquivo(self, revisao: dict) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Escolher nova fotografia", "",
            "Imagens (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)",
        )
        if path:
            aplicar_refoto(self, self.pipeline, revisao, path, on_aplicado=self._carregar)

    def _resolver(self, revisao_id: int) -> None:
        self.repo.resolver_revisao(revisao_id)
        self._carregar()

    def _concluir(self) -> None:
        if self._revisoes:
            return
        resposta = QMessageBox.question(
            self, "Concluir conferencia",
            "Marcar este livro como conferido? Isso encerra a conferencia "
            "e nao impede novas capturas caso o livro seja reaberto.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if resposta == QMessageBox.StandardButton.Yes:
            self.repo.marcar_livro_conferido(self.livro_id)
            self._carregar()
            self.accept()
