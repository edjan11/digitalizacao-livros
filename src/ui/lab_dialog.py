from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QTextEdit, QFrame, QListWidget, QListWidgetItem, QSplitter, QProgressBar,
    QMessageBox,
)

import cv2
from ..ocr.engines import TesseractProvider, RapidOCRProvider
from ..ocr.htr_engine import HTREngine
from ..ocr.combiner import OCRCombiner

logger = logging.getLogger(__name__)


class LabWorker(QThread):
    progresso = Signal(int, int, str)
    resultado = Signal(str, dict)

    def __init__(self, imagens: list[Path]) -> None:
        super().__init__()
        self.imagens = imagens

    def run(self) -> None:
        combiner = OCRCombiner()
        tess = TesseractProvider()
        rapid = RapidOCRProvider()
        htr = HTREngine()
        if tess.is_available():
            combiner.add_provider(tess)
        if rapid.is_available():
            combiner.add_provider(rapid)
        if htr.is_available():
            combiner.add_provider(htr)
        total = len(self.imagens)
        for idx, img_path in enumerate(self.imagens):
            self.progresso.emit(idx + 1, total, img_path.name)
            image = cv2.imread(str(img_path))
            if image is None:
                self.resultado.emit(str(img_path), {"erro": "Nao foi possivel ler"})
                continue
            results = combiner.recognize_all(image)
            termo = combiner.extrair_termo(results)
            folha = combiner.extrair_folha(results)
            tempos = {r.motor: round(r.tempo_ms, 1) for r in results if r.tempo_ms > 0}
            textos = {r.motor: r.texto_bruto[:200] for r in results}
            self.resultado.emit(str(img_path), {
                "termo": termo.valor,
                "termo_status": termo.status,
                "termo_confianca": round(termo.confianca, 3),
                "motor_termo": termo.motor_principal,
                "folha": folha.valor,
                "folha_status": folha.status,
                "tempos": tempos,
                "textos": textos,
            })


class LabDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Laboratorio - Teste de OCR/HTR")
        self.setMinimumSize(1100, 700)
        self._worker: LabWorker | None = None
        self._resultados: dict[str, dict] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        btn_pasta = QPushButton("Selecionar Pasta com Imagens")
        btn_pasta.setMinimumHeight(40)
        btn_pasta.setStyleSheet("QPushButton { background: #1976d2; color: white; font-weight: bold; border-radius: 6px; font-size: 13px; }")
        btn_pasta.clicked.connect(self._selecionar_pasta)
        top.addWidget(btn_pasta)
        btn_limpar = QPushButton("Limpar")
        btn_limpar.clicked.connect(self._limpar)
        top.addWidget(btn_limpar)
        layout.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.lista = QListWidget()
        self.lista.currentItemChanged.connect(self._on_item_selected)
        splitter.addWidget(self.lista)

        self.detalhes = QTextEdit()
        self.detalhes.setReadOnly(True)
        self.detalhes.setFont(QFont("Consolas", 10))
        splitter.addWidget(self.detalhes)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def _selecionar_pasta(self) -> None:
        pasta = QFileDialog.getExistingDirectory(self, "Selecionar pasta com imagens")
        if not pasta:
            return
        imagens = list(Path(pasta).glob("*.jpg")) + list(Path(pasta).glob("*.jpeg")) + list(Path(pasta).glob("*.png"))
        if not imagens:
            QMessageBox.information(self, "Info", "Nenhuma imagem encontrada na pasta.")
            return
        imagens.sort()
        self._limpar()
        self.progress.setVisible(True)
        self.progress.setMaximum(len(imagens))
        self._resultados = {}
        for img in imagens:
            item = QListWidgetItem(img.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img))
            self.lista.addItem(item)
        self._worker = LabWorker(imagens)
        self._worker.progresso.connect(self._on_progress)
        self._worker.resultado.connect(self._on_result)
        self._worker.finished.connect(lambda: self.progress.setVisible(False))
        self._worker.start()

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, name: str) -> None:
        self.progress.setValue(current)
        self.progress.setFormat(f"{current}/{total} - {name}")

    @Slot(str, dict)
    def _on_result(self, path: str, data: dict) -> None:
        self._resultados[path] = data

    @Slot()
    def _on_item_selected(self, current, previous) -> None:
        if not current:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        data = self._resultados.get(path, {})
        if not data:
            self.detalhes.setText("Processando...")
            return
        texto = f"Arquivo: {Path(path).name}\n"
        texto += "=" * 50 + "\n\n"
        if "erro" in data:
            texto += f"ERRO: {data['erro']}\n"
        else:
            texto += f"TERMO: {data.get('termo', '?')}\n"
            texto += f"  Status: {data.get('termo_status', '?')}\n"
            texto += f"  Confianca: {data.get('termo_confianca', '?')}\n"
            texto += f"  Motor: {data.get('motor_termo', '?')}\n\n"
            texto += f"FOLHA: {data.get('folha', '?')}\n"
            texto += f"  Status: {data.get('folha_status', '?')}\n\n"
            tempos = data.get("tempos", {})
            texto += "TEMPOS (ms):\n"
            for motor, t in tempos.items():
                texto += f"  {motor}: {t}ms\n"
            texto += "\nTEXTOS:\n"
            textos = data.get("textos", {})
            for motor, t in textos.items():
                texto += f"  [{motor}]: {t}\n"
        self.detalhes.setText(texto)

    def _limpar(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self.lista.clear()
        self.detalhes.clear()
        self._resultados = {}
        self.progress.setVisible(False)
