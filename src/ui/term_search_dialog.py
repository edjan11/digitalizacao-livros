from __future__ import annotations

from pathlib import Path

import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..database.repository import Repository
from .image_viewer import ImageViewer


def posicao_do_termo(termo: int, termo_inicial: int, termo_final: int) -> tuple[int, int]:
    total = max(1, termo_final - termo_inicial + 1)
    indice = max(0, min(termo - termo_inicial, total - 1))
    return indice, total


class TermSearchDialog(QDialog):
    def __init__(self, repo: Repository, livro_id: int, parent=None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.livro_id = livro_id
        self.setWindowTitle("Localizar registro por termo")
        self.setMinimumSize(1000, 760)
        self._destaque_atual: dict | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Localizar registro por termo")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        ajuda = QLabel(
            "A foto inteira sera exibida e a moldura indicara qual registro contem o termo."
        )
        ajuda.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ajuda.setStyleSheet("color: #616161;")
        layout.addWidget(ajuda)

        busca = QHBoxLayout()
        busca.addWidget(QLabel("Numero do termo:"))
        self.termo = QSpinBox()
        self.termo.setRange(1, 9_999_999)
        self.termo.setMinimumHeight(38)
        self.termo.setFont(QFont("Segoe UI", 13))
        self.termo.lineEdit().returnPressed.connect(self._buscar)
        busca.addWidget(self.termo, 1)
        btn = QPushButton("BUSCAR")
        btn.setMinimumHeight(38)
        btn.setStyleSheet(
            "QPushButton { background: #1976d2; color: white; font-weight: bold; "
            "border-radius: 5px; padding: 4px 24px; }"
        )
        btn.clicked.connect(self._buscar)
        busca.addWidget(btn)
        layout.addLayout(busca)

        self.info = QLabel("Digite um termo para localizar a fotografia.")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setWordWrap(True)
        self.info.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(self.info)

        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)

        zoom = QHBoxLayout()
        zoom.addStretch()
        menos = QPushButton("−")
        menos.setToolTip("Diminuir zoom")
        menos.clicked.connect(self.viewer.zoom_out)
        zoom.addWidget(menos)
        ajustar = QPushButton("Ajustar")
        ajustar.clicked.connect(self.viewer.fit_to_window)
        zoom.addWidget(ajustar)
        cem = QPushButton("100%")
        cem.setToolTip("Um pixel da foto por pixel da tela")
        cem.clicked.connect(self.viewer.zoom_100)
        zoom.addWidget(cem)
        mais = QPushButton("+")
        mais.setToolTip("Aumentar zoom")
        mais.clicked.connect(self.viewer.zoom_in)
        zoom.addWidget(mais)
        zoom.addStretch()
        layout.addLayout(zoom)

        fechar = QPushButton("Fechar")
        fechar.clicked.connect(self.accept)
        layout.addWidget(fechar)
        self.termo.setFocus()

    def _buscar(self) -> None:
        termo = self.termo.value()
        imagem = self.repo.buscar_imagem_por_termo(self.livro_id, termo)
        if not imagem:
            self.viewer.clear()
            self.info.setText(f"Termo {termo} nao encontrado neste livro.")
            QMessageBox.information(self, "Termo nao encontrado", self.info.text())
            return

        termo_inicial = int(imagem["termo_inicial"])
        termo_final = int(imagem["termo_final"])
        indice, total = posicao_do_termo(termo, termo_inicial, termo_final)
        path = Path(imagem.get("caminho_original") or "")
        faixa = str(termo_inicial) if termo_inicial == termo_final else f"{termo_inicial}-{termo_final}"
        folha = imagem.get("folha_estimada")
        face = (imagem.get("face") or "indeterminada").capitalize()
        posicao = indice + 1
        self.info.setText(
            f"Termo {termo} | registro {posicao} de {total} | "
            f"foto {faixa} | folha {folha or '?'} - {face} | {path.name}"
        )
        self._destaque_atual = {
            "path": path,
            "indice": indice,
            "total": total,
            "texto": f"TERMO {termo} - REGISTRO {posicao}/{total}",
            "rotacao": int(imagem.get("rotacao_visualizacao") or 0),
        }
        self._reexibir_atual()

    def _reexibir_atual(self) -> None:
        if not self._destaque_atual:
            return
        atual = self._destaque_atual
        path = atual["path"]
        rotacao = int(atual.get("rotacao") or 0)
        if rotacao in (90, 180, 270):
            imagem = cv2.imread(str(path))
            if imagem is not None:
                if rotacao == 180:
                    imagem = cv2.rotate(imagem, cv2.ROTATE_180)
                elif rotacao == 90:
                    imagem = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
                else:
                    imagem = cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.viewer.set_image_array(
                    imagem,
                    destaque_indice=atual["indice"],
                    total_registros=atual["total"],
                    texto_destaque=atual["texto"],
                )
                return
        self.viewer.set_image_path(
            path,
            destaque_indice=atual["indice"],
            total_registros=atual["total"],
            texto_destaque=atual["texto"],
        )
