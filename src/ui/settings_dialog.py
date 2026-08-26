from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QFormLayout,
    QDialogButtonBox, QGroupBox, QWidget,
)

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Tela para personalizar pastas e parametros operacionais.

    Grava diretamente no config.yaml (Settings.save) e, como o MainWindow
    mantem a mesma instancia de Settings em memoria, as novas pastas passam a
    valer ja na proxima digitalizacao.
    """

    # (rotulo, secao, chave, tipo)
    CAMPOS = [
        ("Pasta de destino do acervo", "acervo", "root_path", "dir"),
        ("Pasta monitorada (scanner CZUR)", "czur", "watch_folder", "dir"),
        ("Debounce da pasta (ms)", "czur", "debounce_ms", "int"),
        ("Indice da camera", "camera", "index", "int"),
        ("DPI de armazenamento", "imaging", "storage_dpi", "int"),
        ("Qualidade JPEG de armazenamento", "imaging", "storage_jpeg_quality", "int"),
    ]
    CAMPOS_BOOL = [
        ("Melhoria automatica de pagina (estilo scan)", "imaging", "enhance_enabled"),
        ("OCR rapido (RapidOCR)", "ocr", "rapidocr_enabled"),
        ("OCR Tesseract", "ocr", "tesseract_enabled"),
        ("HTR (Qwen/GOT) - lento", "ocr", "htr_enabled"),
    ]
    # Quanto maiores blur/dark/estouro, menos fotos marcadas para revisao.
    CAMPOS_QUALIDADE = [
        ("Limite de foco (blur)", "quality", "blur_threshold", "int"),
        ("Limite de escuridao", "quality", "dark_threshold", "int"),
        ("% estouro de luz tolerado", "quality", "overexposed_pct", "int"),
        ("Inclinacao maxima (graus)", "quality", "skew_max_degrees", "float"),
    ]

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Configuracoes")
        self.setMinimumWidth(560)
        self._widgets: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        geral = QGroupBox("Pastas e captura")
        form = QFormLayout(geral)
        for rotulo, secao, chave, tipo in self.CAMPOS:
            val = self.settings.get(secao, chave, "")
            if tipo == "dir":
                edit = QLineEdit(str(val))
                btn = QPushButton("...")
                btn.setMaximumWidth(32)
                btn.clicked.connect(lambda _=False, e=edit: self._escolher_pasta(e))
                linha = QHBoxLayout()
                linha.addWidget(edit)
                linha.addWidget(btn)
                container = QWidget()
                container.setLayout(linha)
                form.addRow(QLabel(rotulo), container)
            else:
                edit = QSpinBox()
                edit.setRange(0, 100000)
                try:
                    edit.setValue(int(val))
                except (TypeError, ValueError):
                    edit.setValue(0)
                form.addRow(QLabel(rotulo), edit)
            self._widgets[(secao, chave)] = (edit, tipo)
        layout.addWidget(geral)

        qualidade = QGroupBox("Sensibilidade da revisao")
        form_q = QFormLayout(qualidade)
        for rotulo, secao, chave, tipo in self.CAMPOS_QUALIDADE:
            val = self.settings.get(secao, chave, "")
            if tipo == "float":
                edit = QDoubleSpinBox()
                edit.setRange(0.0, 180.0)
                edit.setDecimals(1)
                edit.setSingleStep(0.5)
                try:
                    edit.setValue(float(val))
                except (TypeError, ValueError):
                    edit.setValue(3.0)
            else:
                edit = QSpinBox()
                edit.setRange(1, 10000)
                try:
                    edit.setValue(int(float(val)))
                except (TypeError, ValueError):
                    edit.setValue(1)
            form_q.addRow(QLabel(rotulo), edit)
            self._widgets[(secao, chave)] = (edit, tipo)
        layout.addWidget(qualidade)

        ocr = QGroupBox("OCR / melhorias")
        vbox = QVBoxLayout(ocr)
        for rotulo, secao, chave in self.CAMPOS_BOOL:
            val = bool(self.settings.get(secao, chave, False))
            chk = QCheckBox(rotulo)
            chk.setChecked(val)
            vbox.addWidget(chk)
            self._widgets[(secao, chave)] = (chk, "bool")
        layout.addWidget(ocr)

        aviso = QLabel(
            "Pastas e limiares valem a partir da proxima digitalizacao. "
            "'Melhoria automatica' recorta a pagina, endireita e clareia o fundo."
        )
        aviso.setWordWrap(True)
        aviso.setStyleSheet("color: #616161; font-size: 11px;")
        layout.addWidget(aviso)

        botoes = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        botoes.accepted.connect(self._salvar)
        botoes.rejected.connect(self.reject)
        layout.addWidget(botoes)

    def _escolher_pasta(self, edit: QLineEdit) -> None:
        atual = edit.text() or str(Path.home())
        pasta = QFileDialog.getExistingDirectory(self, "Selecione a pasta", atual)
        if pasta:
            edit.setText(pasta)

    def _salvar(self) -> None:
        for (secao, chave), (widget, tipo) in self._widgets.items():
            if tipo == "dir":
                self.settings.set(secao, chave, widget.text())
            elif tipo == "float":
                self.settings.set(secao, chave, round(widget.value(), 2))
            elif tipo == "int":
                self.settings.set(secao, chave, widget.value())
            elif tipo == "bool":
                self.settings.set(secao, chave, widget.isChecked())
        try:
            self.settings.save()
            logger.info("Configuracoes salvas em %s", self.settings.path)
        except Exception:
            logger.exception("Falha ao salvar config")
        self.accept()
