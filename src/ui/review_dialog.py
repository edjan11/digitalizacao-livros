from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
    QWidget, QFileDialog, QMessageBox,
)

from ..database.repository import Repository
from ..services.scan_pipeline import ScanPipeline
from .camera_capture_dialog import CameraCaptureDialog

logger = logging.getLogger(__name__)


class ReviewDialog(QDialog):
    def __init__(self, repo: Repository, pipeline: ScanPipeline | None = None, parent=None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.pipeline = pipeline
        self._camera_refoto: CameraCaptureDialog | None = None
        self.setWindowTitle("Revisao de Pendentes")
        self.setMinimumSize(700, 500)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Fotos a refazer e itens para revisar")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        self._scroll = QScrollArea()
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll)

        btn_fechar = QPushButton("Fechar")
        btn_fechar.clicked.connect(self.accept)
        layout.addWidget(btn_fechar)

        self._carregar()

    def _carregar(self) -> None:
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        pendentes = self.repo.listar_revisoes_pendentes()
        if not pendentes:
            lbl = QLabel("Nenhuma pendencia para revisar")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Segoe UI", 12))
            self._container_layout.addWidget(lbl)
            self._container_layout.addStretch()
            return
        for rev in pendentes:
            card = self._criar_card(rev)
            self._container_layout.addWidget(card)
        self._container_layout.addStretch()

    def _criar_card(self, revisao: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet("QFrame { border: 1px solid #e0e0e0; border-radius: 6px; background: white; margin: 4px; }")
        layout = QVBoxLayout(card)

        tipo = revisao.get("tipo", "desconhecido")
        tipo_nomes = {
            "refazer_captura": "REFAZER FOTOGRAFIA",
            "duplicidade": "Possivel duplicidade",
            "qualidade": "Problema de qualidade",
            "termo_incerto": "Termo incerto",
            "folha_incerta": "Folha incerta",
            "ocr_falha": "OCR não conseguiu ler",
            "classificar_documento": "Confirmar tipo do documento",
            "nome_incerto": "NOME INCERTO — corrigir com Qwen",
        }
        if tipo == "refazer_captura":
            folha = revisao.get("folha_estimada") or "?"
            face = (revisao.get("face") or "indeterminada").capitalize()
            termo_i = revisao.get("termo_inicial")
            termo_f = revisao.get("termo_final")
            faixa = "?" if termo_i is None else (
                str(termo_i) if termo_i == termo_f else f"{termo_i}-{termo_f}"
            )
            nome_tipo = f"REFAZER - Folha {folha} - {face} - Termos {faixa}"
        else:
            nome_tipo = tipo_nomes.get(tipo, tipo)

        header = QLabel(nome_tipo)
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        color = {
            "refazer_captura": "#d84315",
            "duplicidade": "#f44336",
            "qualidade": "#ff9800",
            "termo_incerto": "#2196f3",
            "folha_incerta": "#9c27b0",
            "ocr_falha": "#c62828",
            "classificar_documento": "#6a1b9a",
            "nome_incerto": "#6a1b9a",
        }.get(tipo, "#757575")
        header.setStyleSheet(f"color: {color};")
        layout.addWidget(header)

        thumb = Path(revisao.get("caminho_thumb") or "")
        original = Path(revisao.get("caminho_original") or "")
        imagem_path = thumb if thumb.is_file() else original
        if imagem_path.is_file():
            pix = QPixmap(str(imagem_path))
            if not pix.isNull():
                imagem = QLabel()
                imagem.setAlignment(Qt.AlignmentFlag.AlignCenter)
                imagem.setPixmap(
                    pix.scaled(
                        560, 230,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                layout.addWidget(imagem)

        detalhes = QLabel(revisao.get("detalhes", ""))
        detalhes.setWordWrap(True)
        layout.addWidget(detalhes)

        btn_layout = QHBoxLayout()
        if tipo == "refazer_captura" and self.pipeline is not None:
            btn_camera = QPushButton("Refotografar com camera")
            btn_camera.clicked.connect(lambda: self._abrir_camera_refoto(revisao))
            btn_layout.addWidget(btn_camera)
            btn_arquivo = QPushButton("Escolher nova foto")
            btn_arquivo.clicked.connect(lambda: self._escolher_refoto(revisao))
            btn_layout.addWidget(btn_arquivo)
        else:
            btn_ok = QPushButton("OK - Confirmar")
            btn_ok.clicked.connect(lambda: self._resolver(revisao["id"]))
            btn_layout.addWidget(btn_ok)

        btn_ignorar = QPushButton("Ignorar (nao e problema)")
        btn_ignorar.clicked.connect(lambda: self._resolver(revisao["id"]))
        btn_layout.addWidget(btn_ignorar)
        layout.addLayout(btn_layout)

        return card

    def _resolver(self, revisao_id: int) -> None:
        self.repo.resolver_revisao(revisao_id)
        self._carregar()

    @staticmethod
    def _contexto_refoto(revisao: dict) -> str:
        folha = revisao.get("folha_estimada") or "?"
        face = (revisao.get("face") or "indeterminada").capitalize()
        termo_i, termo_f = revisao.get("termo_inicial"), revisao.get("termo_final")
        faixa = "?" if termo_i is None else (
            str(termo_i) if termo_i == termo_f else f"{termo_i}-{termo_f}"
        )
        return f"REFAZER: folha {folha} - {face} - termos {faixa}"

    def _abrir_camera_refoto(self, revisao: dict) -> None:
        if self.pipeline is None:
            return
        livro_id = revisao.get("livro_id")
        pasta = self.pipeline.acervo_root / f"livro_{livro_id}" / "refotos_camera"
        indice = self.pipeline.settings.get("camera", "index", 0) if self.pipeline.settings else 0
        dlg = CameraCaptureDialog(
            pasta,
            lambda: self._contexto_refoto(revisao),
            indice,
            self,
        )
        self._camera_refoto = dlg
        dlg.foto_capturada.connect(
            lambda path: self._receber_refoto_camera(revisao, path, dlg)
        )
        dlg.exec()
        self._camera_refoto = None

    def _receber_refoto_camera(
        self, revisao: dict, path: str, dialog: CameraCaptureDialog
    ) -> None:
        dialog.accept()
        self._aplicar_refoto(revisao, path)

    def _escolher_refoto(self, revisao: dict) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Escolher nova fotografia",
            "",
            "Imagens (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)",
        )
        if path:
            self._aplicar_refoto(revisao, path)

    def _aplicar_refoto(self, revisao: dict, path: str) -> None:
        if self.pipeline is None:
            return
        resultado = self.pipeline.substituir_captura(
            revisao["imagem_id"], revisao["id"], path
        )
        if resultado.get("erro"):
            QMessageBox.warning(self, "Nao foi possivel substituir", resultado["erro"])
        elif resultado.get("substituida"):
            ocr = self.pipeline.processar_ocr_secundario(resultado["imagem_id"])
            complemento = (
                "\nO OCR da nova fotografia foi processado e salvo."
                if not ocr.get("erro")
                else f"\nA fotografia foi salva, mas o OCR apontou erro: {ocr['erro']}"
            )
            QMessageBox.information(
                self,
                "Fotografia substituida",
                f"{self._contexto_refoto(revisao)} foi corrigida sem alterar a contagem."
                f"{complemento}",
            )
        else:
            motivos = ", ".join(resultado["qualidade"].get("motivos_refazer", []))
            QMessageBox.warning(
                self,
                "A nova foto ainda precisa ser refeita",
                f"Ela continuara na lista: {motivos}",
            )
        self._carregar()
