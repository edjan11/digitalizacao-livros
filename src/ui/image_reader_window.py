from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..imaging.record_regions import bbox_registro
from ..ocr.combiner import texto_evidencia_termo

from .image_viewer import ImageViewer
from .theme import STATUS_OK, STATUS_ERRO, TEXTO_NEON


TIPOS_REVISOR = {
    "nome_registrado": "Nome do registrado",
    "nome_mae": "Nome da mãe",
    "declarante": "Declarante",
    "pai_ou_mae": "Filiação",
    "data": "Data",
    "ano": "Ano",
    "termo": "Termo",
    "folha": "Folha",
    "local": "Local",
    "texto_linha": "Texto OCR",
}


class ImageReaderWindow(QMainWindow):
    """Janela completa para revisão; a consulta principal permanece leve."""

    qwen_requested = Signal(object, object)
    qwen_record_requested = Signal(object, object)

    def __init__(
        self,
        *,
        image_path: Path,
        image: np.ndarray,
        registro: dict,
        metadados: list[dict],
        repo=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.image_path = Path(image_path)
        self.image = image.copy()
        self.registro = dict(registro)
        self._metadados = list(metadados)
        self.repo = repo
        self.setWindowTitle(
            f"Revisor — termo {self.registro.get('termo') or '?'} — "
            f"{self.registro.get('livro_codigo') or '?'}"
        )
        self.setMinimumSize(1100, 720)
        self.resize(1500, 900)
        self._montar_ui()

    def _montar_ui(self) -> None:
        raiz = QWidget(self)
        layout = QVBoxLayout(raiz)
        layout.setContentsMargins(8, 8, 8, 8)

        cabecalho = QHBoxLayout()
        contexto = QLabel(
            f"Termo {self.registro.get('termo') or '?'} | "
            f"Livro {self.registro.get('livro_codigo') or '?'} | "
            f"Folha {self.registro.get('folha') or '?'} — "
            f"{(self.registro.get('face') or '?').capitalize()}"
        )
        contexto.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        cabecalho.addWidget(contexto)
        cabecalho.addStretch()
        evidencia = texto_evidencia_termo(self.registro)
        if evidencia:
            evid = QLabel(evidencia)
            evid.setStyleSheet("color:#6a1b9a; font-size:11px;")
            evid.setWordWrap(True)
            evid.setMaximumWidth(340)
            cabecalho.addWidget(evid)
        self.status = QLabel(self._status_texto())
        self.status.setStyleSheet(self._status_estilo())
        cabecalho.addWidget(self.status)
        layout.addLayout(cabecalho)

        split = QSplitter(Qt.Orientation.Horizontal)
        painel_foto = QWidget()
        foto_layout = QVBoxLayout(painel_foto)
        self.viewer = ImageViewer()
        self.viewer.set_selection_mode(False)
        self.viewer.set_image_array(
            self.image,
            destaque_indice=int(self.registro.get("indice_na_imagem") or 0),
            total_registros=self._total_registros(),
            texto_destaque=f"Termo {self.registro.get('termo') or '?'}",
        )
        self.viewer.selection_changed.connect(self._area_selecionada)
        foto_layout.addWidget(self.viewer, 1)
        split.addWidget(painel_foto)

        painel_info = QWidget()
        info_layout = QVBoxLayout(painel_info)
        info_layout.addWidget(QLabel("Transcrição e metadados"))
        self.transcricao = QTextEdit()
        self.transcricao.setReadOnly(True)
        self.transcricao.setFont(QFont("Consolas", 10))
        self.transcricao.setPlainText(self._texto_transcricao())
        info_layout.addWidget(self.transcricao, 2)
        self.tabela_metadados = QTableWidget(0, 6)
        self.tabela_metadados.setHorizontalHeaderLabels(
            ["Campo", "Valor", "Conf.", "Motor", "Situação", "Escopo"]
        )
        self.tabela_metadados.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabela_metadados.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabela_metadados.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.tabela_metadados.verticalHeader().setVisible(False)
        self.tabela_metadados.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tabela_metadados.cellDoubleClicked.connect(lambda _r, _c: self._corrigir_metadado())
        self._preencher_metadados()
        info_layout.addWidget(self.tabela_metadados, 2)
        acoes_meta = QHBoxLayout()
        self._botao("Adicionar metadado", self._adicionar_metadado, acoes_meta)
        self._botao("Corrigir valor", self._corrigir_metadado, acoes_meta)
        self._botao("Confirmar detecção", self._confirmar_metadado, acoes_meta)
        acoes_meta.addStretch()
        info_layout.addLayout(acoes_meta)
        split.addWidget(painel_info)
        split.setStretchFactor(0, 7)
        split.setStretchFactor(1, 3)
        layout.addWidget(split, 1)

        controles = QHBoxLayout()
        self._botao("Foto inteira", self.viewer.fit_to_window, controles)
        self._botao("Ajustar largura", self.viewer.fit_to_width, controles)
        for texto, valor in (("50%", 50), ("100%", 100), ("150%", 150), ("200%", 200)):
            self._botao(texto, lambda v=valor: self.viewer.set_zoom_percent(v), controles)
        self._botao("Registro de cima", lambda: self._focar(0), controles)
        self._botao("Registro de baixo", lambda: self._focar(1), controles)
        self.btn_selecionar = QPushButton("Selecionar área")
        self.btn_selecionar.setCheckable(True)
        self.btn_selecionar.toggled.connect(self.viewer.set_selection_mode)
        controles.addWidget(self.btn_selecionar)
        self.btn_qwen = QPushButton("Ler área com Qwen")
        self.btn_qwen.setEnabled(False)
        self.btn_qwen.clicked.connect(self._pedir_qwen)
        controles.addWidget(self.btn_qwen)
        self.btn_qwen_registro = QPushButton("Qwen: nome, termo e mãe")
        self.btn_qwen_registro.setToolTip(
            "Uma única leitura do registro selecionado: nome, número do termo e mãe."
        )
        self.btn_qwen_registro.clicked.connect(self._pedir_qwen_registro)
        self.btn_qwen_registro.setText("Qwen: nome, termo, mae e data")
        self.btn_qwen_registro.setToolTip(
            "Uma unica leitura do registro selecionado: nome, termo, mae e data do registro."
        )
        controles.addWidget(self.btn_qwen_registro)
        controles.addStretch()
        self._botao("Tela cheia", self._alternar_tela_cheia, controles)
        self._botao("Abrir original no Windows", self._abrir_externo, controles)
        self._botao("Copiar caminho", self._copiar_caminho, controles)
        layout.addLayout(controles)
        self.setCentralWidget(raiz)

    @staticmethod
    def _botao(texto: str, acao, layout: QHBoxLayout) -> QPushButton:
        botao = QPushButton(texto)
        botao.clicked.connect(acao)
        layout.addWidget(botao)
        return botao

    def _total_registros(self) -> int:
        inicial = self.registro.get("termo_inicial")
        final = self.registro.get("termo_final")
        if inicial is not None and final is not None:
            return max(1, int(final) - int(inicial) + 1)
        return 1

    def _focar(self, indice: int) -> None:
        self.viewer.set_highlight(
            indice, self._total_registros(),
        )
        self.viewer.focus_highlight()

    def _area_selecionada(self) -> None:
        self.btn_qwen.setEnabled(self.viewer.selected_relative_rect() is not None)

    def _pedir_qwen(self) -> None:
        bbox = self.viewer.selected_relative_rect()
        if bbox is None:
            QMessageBox.information(self, "Qwen", "Selecione uma área primeiro.")
            return
        self.qwen_requested.emit(self.image.copy(), bbox)

    def _bbox_registro(self) -> tuple[float, float, float, float]:
        """Área do assento compartilhada por visualizador e OCR."""
        total = self._total_registros()
        indice = max(0, min(int(self.registro.get("indice_na_imagem") or 0), total - 1))
        return bbox_registro(indice, total)

    def _pedir_qwen_registro(self) -> None:
        self.qwen_record_requested.emit(self.image.copy(), self._bbox_registro())

    def _alternar_tela_cheia(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _abrir_externo(self) -> None:
        if not self.image_path.is_file():
            QMessageBox.warning(self, "Imagem", "O arquivo original não foi encontrado.")
            return
        try:
            os.startfile(str(self.image_path))
        except (AttributeError, OSError) as exc:
            QMessageBox.warning(self, "Imagem", f"Não foi possível abrir no Windows: {exc}")

    def _copiar_caminho(self) -> None:
        QApplication.clipboard().setText(str(self.image_path))
        self.status.setText("CAMINHO COPIADO")
        self.status.setStyleSheet(f"color:{TEXTO_NEON}; font-weight:bold;")

    def _status_texto(self) -> str:
        statuses = [str(m.get("status") or "") for m in self._metadados]
        if any("confirm" in s.lower() or "corrig" in s.lower() for s in statuses):
            return "CONFIRMADO"
        if any("revis" in s.lower() or "duvid" in s.lower() for s in statuses):
            return "REVISAR"
        if self._metadados:
            return "INFORMAÇÃO"
        return "SEM OCR"

    def _status_estilo(self) -> str:
        texto = self._status_texto()
        cor = STATUS_OK if texto == "CONFIRMADO" else STATUS_ERRO if texto == "REVISAR" else TEXTO_NEON
        return f"color:{cor}; font-weight:bold; padding:4px 8px;"

    def _texto_transcricao(self) -> str:
        integrais = [
            m for m in self._metadados if m.get("tipo") == "transcricao_integral"
        ]
        if integrais:
            return "\n\n".join(
                str(m.get("valor_tratado") or m.get("valor_original") or "")
                for m in integrais
            )
        return "Transcrição integral ainda não processada.\n\nUse o revisor para ampliar, selecionar uma área ou iniciar a leitura pendente."

    def _preencher_metadados(self) -> None:
        self.tabela_metadados.setRowCount(len(self._metadados))
        for linha, meta in enumerate(self._metadados):
            valores = [
                TIPOS_REVISOR.get(meta.get("tipo"), meta.get("tipo") or "?"),
                meta.get("valor_tratado") or meta.get("valor_original") or "",
                f"{float(meta.get('confianca') or 0) * 100:.0f}%",
                meta.get("motor") or meta.get("fonte") or "—",
                meta.get("status") or "—",
                meta.get("escopo") or "—",
            ]
            for coluna, valor in enumerate(valores):
                item = QTableWidgetItem(str(valor))
                if coluna == 0 and meta.get("id") is not None:
                    item.setData(Qt.ItemDataRole.UserRole, meta["id"])
                self.tabela_metadados.setItem(linha, coluna, item)

    def atualizar_metadados(self, metadados: list[dict]) -> None:
        """Atualiza a coluna lateral sem recarregar a foto de alta resolução."""
        self._metadados = list(metadados)
        self.transcricao.setPlainText(self._texto_transcricao())
        self._preencher_metadados()
        self.status.setText(self._status_texto())
        self.status.setStyleSheet(self._status_estilo())

    def _deteccao_selecionada(self) -> tuple[int | None, int]:
        linha = self.tabela_metadados.currentRow()
        if linha < 0:
            return None, linha
        item = self.tabela_metadados.item(linha, 0)
        valor = item.data(Qt.ItemDataRole.UserRole) if item else None
        return (int(valor) if valor is not None else None), linha

    def _adicionar_metadado(self) -> None:
        if self.repo is None:
            return
        nomes = list(TIPOS_REVISOR.values())[:-1]
        rotulo, ok = QInputDialog.getItem(self, "Campo", "Tipo do metadado:", nomes, 0, False)
        if not ok:
            return
        valor, ok = QInputDialog.getText(self, "Valor", f"{rotulo}:")
        if not ok or not valor.strip():
            return
        tipo = next(chave for chave, nome in TIPOS_REVISOR.items() if nome == rotulo)
        self.repo.salvar_metadado_tratado(
            imagem_id=self.registro["imagem_id"],
            registro_id=self.registro["registro_id"],
            tipo=tipo,
            valor=valor.strip(),
            confianca=1.0,
            fonte="operador",
            motor="operador",
            status="confirmado",
        )
        self.atualizar_metadados(self.repo.listar_metadados_registro(self.registro["registro_id"]))

    def _corrigir_metadado(self) -> None:
        if self.repo is None:
            return
        deteccao_id, linha = self._deteccao_selecionada()
        if deteccao_id is None or linha < 0:
            return
        atual = self.tabela_metadados.item(linha, 1).text()
        novo, ok = QInputDialog.getText(self, "Corrigir metadado", "Valor correto:", text=atual)
        if ok and novo.strip() and novo.strip() != atual:
            self.repo.corrigir_deteccao(deteccao_id, novo.strip())
            self.atualizar_metadados(self.repo.listar_metadados_registro(self.registro["registro_id"]))

    def _confirmar_metadado(self) -> None:
        if self.repo is None:
            return
        deteccao_id, _linha = self._deteccao_selecionada()
        if deteccao_id is not None:
            self.repo.confirmar_deteccao(deteccao_id)
            self.atualizar_metadados(self.repo.listar_metadados_registro(self.registro["registro_id"]))
