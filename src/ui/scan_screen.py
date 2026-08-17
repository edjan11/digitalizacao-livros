from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGroupBox,
    QProgressBar, QMessageBox,
)

from ..database.repository import Repository
from ..session.scan_session import ScanSession
from ..services.scan_pipeline import ScanPipeline
from ..services.telemetry import emitir, registrar_amostrador
from .term_search_dialog import TermSearchDialog
from .camera_capture_dialog import CameraCaptureDialog
from .image_viewer import ImageViewer

logger = logging.getLogger(__name__)


class OCRWorker(QThread):
    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(self, pipeline: ScanPipeline, imagem_id: int) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.imagem_id = imagem_id

    def run(self) -> None:
        try:
            result = self.pipeline.processar_ocr_secundario(self.imagem_id)
            self.concluido.emit(result)
        except Exception as e:
            self.erro.emit(str(e))


class CapturaPipelineWorker(QThread):
    """Persiste uma foto da câmera fora da thread da interface.

    A câmera continua mostrando a prévia e aceitando a próxima folha enquanto
    hashes, qualidade e comparação de duplicidade são calculados.
    """

    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(self, pipeline: ScanPipeline, image_path: str) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path

    def run(self) -> None:
        try:
            self.concluido.emit(self.pipeline.processar_imagem_imediato(self.image_path))
        except Exception as exc:
            logger.exception("Erro no processamento da captura em segundo plano")
            self.erro.emit(str(exc))


class ScanScreen(QWidget):
    voltar_clicked = Signal()
    revisao_clicked = Signal()
    config_clicked = Signal()

    def __init__(self, repo: Repository, session: ScanSession, pipeline: ScanPipeline, parent=None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.session = session
        self.pipeline = pipeline
        self._ultima_imagem_id: int | None = None
        self._ultimo_resultado: dict | None = None
        self._ocr_worker: OCRWorker | None = None
        self._ocr_queue: deque[int] = deque()
        self._captura_worker: CapturaPipelineWorker | None = None
        self._captura_queue: deque[str] = deque()

        def _tamanhos_filas() -> dict:
            return {"fila_ocr": len(self._ocr_queue), "fila_captura": len(self._captura_queue)}

        registrar_amostrador("filas_ui", _tamanhos_filas)
        self._init_ui()
        self._atualizar_pendentes()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        path_text = self.session.resumo
        titulo = QLabel(path_text)
        titulo.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        titulo.setStyleSheet("color: #1565c0;")
        header.addWidget(titulo)
        header.addStretch()
        self.lbl_pendentes = QPushButton("Revisao (0)")
        self.lbl_pendentes.setStyleSheet(
            "QPushButton { background: #ff9800; color: white; border-radius: 4px; padding: 4px 12px; font-weight: bold; } "
            "QPushButton:hover { background: #f57c00; }"
        )
        self.lbl_pendentes.clicked.connect(self.revisao_clicked.emit)
        header.addWidget(self.lbl_pendentes)
        btn_buscar = QPushButton("Buscar termo")
        btn_buscar.setStyleSheet(
            "QPushButton { background: #1976d2; color: white; border-radius: 4px; "
            "padding: 4px 12px; font-weight: bold; }"
        )
        btn_buscar.clicked.connect(self._buscar_termo)
        header.addWidget(btn_buscar)
        btn_camera = QPushButton("Camera automatica")
        btn_camera.setStyleSheet(
            "QPushButton { background: #00897b; color: white; border-radius: 4px; "
            "padding: 4px 12px; font-weight: bold; }"
        )
        btn_camera.clicked.connect(self._abrir_camera_automatica)
        header.addWidget(btn_camera)
        layout.addLayout(header)

        info_bar = QFrame()
        info_bar.setStyleSheet("QFrame { background: #f5f5f5; border-radius: 6px; padding: 8px; }")
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(8, 4, 8, 4)
        info1 = QLabel(f"Folha: {self.session.ultima_folha} - {self.session.ultima_face.capitalize()}" if self.session.ultima_folha else "Aguardando imagens...")
        info1.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        info_layout.addWidget(info1)
        self.lbl_folha_face = info1
        info_layout.addStretch()
        info2 = QLabel(f"Termo: {self.session.ultimo_termo}" if self.session.ultimo_termo else "")
        info2.setFont(QFont("Segoe UI", 12))
        info_layout.addWidget(info2)
        self.lbl_termo = info2
        layout.addWidget(info_bar)

        # A fotografia Ã© o elemento principal da tela. Indicadores e
        # contadores ficam em uma coluna estreita, sem reduzir a Ã¡rea Ãºtil
        # para o operador conferir foco e enquadramento.
        content = QHBoxLayout()
        content.setSpacing(10)

        foto_panel = QFrame()
        foto_panel.setStyleSheet(
            "QFrame { border: 1px solid #607d8b; border-radius: 8px; background: #202124; }"
        )
        foto_layout = QVBoxLayout(foto_panel)
        foto_layout.setContentsMargins(6, 6, 6, 6)
        foto_header = QHBoxLayout()
        foto_title = QLabel("FOTO CAPTURADA — confira o enquadramento")
        foto_title.setStyleSheet("color:#ffffff; font-weight:bold; font-size:13px;")
        foto_header.addWidget(foto_title)
        foto_header.addStretch()
        btn_foto_pagina = QPushButton("Ajustar à página")
        btn_foto_pagina.setToolTip("Mostrar a fotografia inteira")
        btn_foto_pagina.clicked.connect(lambda: self.foto_viewer.fit_to_page())
        foto_header.addWidget(btn_foto_pagina)
        btn_foto_100 = QPushButton("100%")
        btn_foto_100.clicked.connect(self.foto_viewer_zoom_100)
        foto_header.addWidget(btn_foto_100)
        foto_layout.addLayout(foto_header)
        self.foto_viewer = ImageViewer()
        self.foto_viewer.setMinimumSize(620, 430)
        self.foto_viewer.set_selection_mode(False)
        foto_layout.addWidget(self.foto_viewer, 1)
        self.lbl_foto_info = QLabel("Nenhuma fotografia capturada ainda")
        self.lbl_foto_info.setStyleSheet("color:#d9e2ec; font-size:11px;")
        self.lbl_foto_info.setWordWrap(True)
        foto_layout.addWidget(self.lbl_foto_info)
        content.addWidget(foto_panel, 1)

        indicators = QGroupBox("Indicadores")
        indicators.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        ind_layout = QVBoxLayout(indicators)
        ind_layout.setSpacing(6)

        self.ind_imagem = self._make_indicator("Imagem")
        self.ind_foco = self._make_indicator("Foco")
        self.ind_enquadramento = self._make_indicator("Enquadramento")
        self.ind_duplicidade = self._make_indicator("Duplicidade")
        self.ind_ocr = self._make_indicator("OCR")

        ind_layout.addWidget(self.ind_imagem)
        ind_layout.addWidget(self.ind_foco)
        ind_layout.addWidget(self.ind_enquadramento)
        ind_layout.addWidget(self.ind_duplicidade)
        ind_layout.addWidget(self.ind_ocr)
        ind_layout.addStretch()
        indicators.setMinimumWidth(250)
        indicators.setMaximumWidth(310)
        content.addWidget(indicators, 0)
        layout.addLayout(content)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        bottom = QHBoxLayout()
        btn_voltar = QPushButton("Trocar Livro")
        btn_voltar.clicked.connect(self._on_voltar)
        bottom.addWidget(btn_voltar)
        btn_pausar = QPushButton("Pausar")
        btn_pausar.clicked.connect(self._on_pausar)
        bottom.addWidget(btn_pausar)
        bottom.addStretch()
        self.btn_finalizar = QPushButton("Finalizar Livro")
        self.btn_finalizar.setStyleSheet(
            "QPushButton { background: #4caf50; color: white; border-radius: 6px; padding: 6px 16px; font-weight: bold; } "
            "QPushButton:hover { background: #43a047; }"
        )
        self.btn_finalizar.clicked.connect(self._on_finalizar)
        bottom.addWidget(self.btn_finalizar)
        layout.addLayout(bottom)

    def _make_indicator(self, name: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background: white; border: 1px solid #e0e0e0; border-radius: 4px; padding: 4px; }")
        h = QHBoxLayout(frame)
        h.setContentsMargins(8, 2, 8, 2)
        label = QLabel(name + ":")
        label.setFont(QFont("Segoe UI", 10))
        h.addWidget(label)
        h.addStretch()
        status = QLabel("...")
        status.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        h.addWidget(status)
        frame._status_label = status
        return frame

    def _set_indicator(self, frame: QFrame, status: str, value: str = "") -> None:
        label: QLabel = frame._status_label
        color = {
            "ok": "#4caf50", "confirmado": "#4caf50", "provavel": "#7cb342",
            "inferido_sequencia": "#1976d2", "aviso": "#ff9800",
            "duvidoso": "#ff9800", "revisar": "#f44336",
            "precisa_revisao": "#f44336", "erro_grave": "#d32f2f",
        }.get(status, "#9e9e9e")
        text = {
            "ok": "OK", "confirmado": "CONFIRMADO", "provavel": "PROVAVEL",
            "inferido_sequencia": "SEQUENCIA", "aviso": "AVISO",
            "revisar": "REVISAR", "precisa_revisao": "REVISAR",
            "erro_grave": "ERRO",
        }.get(status, status.upper())
        if value:
            text = f"{text}  {value}"
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def foto_viewer_zoom_100(self) -> None:
        self.foto_viewer.zoom_100()

    def processar_nova_imagem(self, image_path: str) -> None:
        # TambÃ©m as imagens vindas da pasta/CZUR entram na mesma fila da
        # cÃ¢mera. Assim uma sÃ©rie de fotos nunca congela a visualizaÃ§Ã£o.
        self._captura_queue.append(str(image_path))
        self._set_indicator(
            self.ind_imagem,
            "aviso",
            f"fila: {len(self._captura_queue) + (1 if self._captura_worker else 0)}",
        )
        self._iniciar_proxima_captura()

    def _aplicar_resultado_captura(self, result: dict) -> None:
        self._ultimo_resultado = result
        self._ultima_imagem_id = result.get("imagem_id")
        self._atualizar_ui_imediatamente(result)
        if result.get("aguarda_confirmacao_duplicidade") and self._ultima_imagem_id:
            # Esta rara pergunta precisa ser resolvida na hora, pois uma
            # página repetida não pode avançar folha/termos.
            self._confirmar_possivel_duplicata(result)
        if result.get("nao_registro"):
            self.progress.setVisible(False)
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(50)
        self.progress.setFormat("OCR em segundo plano...")
        if self._ultima_imagem_id:
            self._enfileirar_ocr(self._ultima_imagem_id)
        else:
            self.progress.setVisible(False)

    def _confirmar_possivel_duplicata(self, result: dict) -> None:
        dup = result.get("duplicidade", {})
        ref = self.repo.get_imagem(dup.get("duplicata_de")) if dup.get("duplicata_de") else None
        ref_nome = Path(ref.get("caminho_original", "")).name if ref else "foto anterior"
        resposta = QMessageBox.question(
            self,
            "Possivel pagina repetida",
            "A escrita desta foto se parece com uma pagina ja capturada.\n\n"
            f"Comparacao: {ref_nome}\n"
            "Esta e a mesma pagina fotografada novamente?\n\n"
            "Se confirmar, o contador de folha/termo nao avancara.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        confirmar = resposta == QMessageBox.StandardButton.Yes
        resolvida = self.pipeline.resolver_possivel_duplicata(
            result["imagem_id"], confirmar
        )
        if confirmar:
            termo_i = resolvida.get("termo_inicial")
            termo_f = resolvida.get("termo_final")
            faixa = str(termo_i) if termo_i == termo_f else f"{termo_i}-{termo_f}"
            self.lbl_termo.setText(f"Duplicata dos termos: {faixa}")
            self._set_indicator(self.ind_duplicidade, "revisar", "CONFIRMADA")
        else:
            self._set_indicator(self.ind_duplicidade, "ok", "NAO E DUPLICATA")
        self._atualizar_pendentes()

    def _atualizar_ui_imediatamente(self, result: dict) -> None:
        self._mostrar_foto_resultado(result)
        q = result.get("qualidade", {})
        if result.get("nao_registro"):
            self._set_indicator(self.ind_imagem, "revisar", "SEM TERMO / CONTAGEM PARADA")
            self._set_indicator(self.ind_ocr, "aviso", "NAO EXECUTADO")
            self.lbl_folha_face.setText("Documento preservado — não é face de registro")
            self.lbl_termo.setText("Nenhum termo atribuído; contador não avançou")
            return
        if q.get("repetir_captura"):
            motivos = ", ".join(q.get("motivos_refazer", []))
            self._set_indicator(self.ind_imagem, "revisar", "REFAZER DEPOIS")
            self.ind_imagem._status_label.setToolTip(motivos)
        else:
            self._set_indicator(self.ind_imagem, q.get("status_geral", "ok"))
        self._set_indicator(self.ind_foco, q.get("foco_status", "ok"))
        self._set_indicator(self.ind_enquadramento, q.get("enquadramento_status", "ok"))
        dup = result.get("duplicidade", {})
        self._set_indicator(self.ind_duplicidade, "revisar" if dup.get("status") != "unico" else "ok")
        self.lbl_folha_face.setText(
            f"Folha: {result.get('folha')} - {result.get('face', '').capitalize()}"
        )
        termo_inicial = result.get("termo_inicial")
        termo_final = result.get("termo_final")
        if termo_inicial is not None:
            faixa = str(termo_inicial) if termo_inicial == termo_final else f"{termo_inicial}-{termo_final}"
            self.lbl_termo.setText(f"Termos desta imagem: {faixa}")
        else:
            self.lbl_termo.setText("")
        self._atualizar_pendentes()

    def _mostrar_foto_resultado(self, result: dict) -> None:
        caminho = result.get("caminho_armazenamento")
        origem = "armazenamento 300 DPI"
        if not caminho:
            caminho = result.get("caminho_original")
            origem = "original"
        if not caminho and result.get("imagem_id"):
            imagem = self.repo.get_imagem(result["imagem_id"])
            caminho = (
                (imagem.get("caminho_armazenamento") or imagem.get("caminho_original"))
                if imagem else None
            )
            if imagem and imagem.get("caminho_armazenamento"):
                origem = "armazenamento 300 DPI"
        if not caminho or not Path(caminho).is_file():
            return
        self.foto_viewer.set_image_path(caminho)
        self.foto_viewer.fit_to_page()
        qualidade = result.get("qualidade", {})
        status = qualidade.get("status_geral") or "foto carregada"
        self.lbl_foto_info.setText(
            f"{Path(caminho).name} — {status} — {origem}. "
            "Use 100% e as barras para conferir as linhas e as bordas da folha."
        )

    def _enfileirar_ocr(self, imagem_id: int) -> None:
        self._ocr_queue.append(imagem_id)
        if self._ocr_worker and self._ocr_worker.isRunning():
            self.progress.setFormat(f"OCR em segundo plano ({len(self._ocr_queue)} na fila)")
            return
        self._iniciar_proximo_ocr()

    def _iniciar_proximo_ocr(self) -> None:
        if not self._ocr_queue:
            self._ocr_worker = None
            self.progress.setVisible(False)
            return
        imagem_id = self._ocr_queue.popleft()
        worker = OCRWorker(self.pipeline, imagem_id)
        self._ocr_worker = worker
        worker.concluido.connect(self._on_ocr_concluido)
        worker.erro.connect(self._on_ocr_erro)
        worker.finished.connect(self._on_ocr_worker_finalizado)
        worker.start()

    @Slot()
    def _on_ocr_worker_finalizado(self) -> None:
        worker = self.sender()
        if worker:
            worker.deleteLater()
        self._ocr_worker = None
        self._iniciar_proximo_ocr()

    @Slot(dict)
    def _on_ocr_concluido(self, result: dict) -> None:
        self.progress.setValue(100)
        self.progress.setFormat(
            "OCR carregado do banco" if result.get("reutilizado") else "OCR processado e salvo"
        )
        termo = result.get("termo")
        termo_inicial = result.get("termo_inicial")
        termo_final = result.get("termo_final")
        status = result.get("termo_status", "")
        conf = result.get("termo_confianca", 0)
        folha = result.get("folha")
        if result.get("imagem_id") == self._ultima_imagem_id:
            if termo_inicial is not None:
                faixa = str(termo_inicial) if termo_inicial == termo_final else f"{termo_inicial}-{termo_final}"
                self.lbl_termo.setText(f"Termos: {faixa} ({status})")
            else:
                self.lbl_termo.setText(f"Termo OCR: {termo or '?'} ({status})")
            if folha:
                self.lbl_folha_face.setText(f"Folha: {folha} - {self.session.ultima_face.capitalize()}")
            self._set_indicator(self.ind_ocr, status)
        motor = result.get("motor", "")
        tempos = result.get("tempos_ms", {})
        self.ind_ocr._status_label.setToolTip(f"Motor: {motor}\nTempos: {tempos}")
        self._atualizar_pendentes()

    @Slot(str)
    def _on_ocr_erro(self, msg: str) -> None:
        self._set_indicator(self.ind_ocr, "erro_grave", msg[:30])
        logger.error("Erro OCR: %s", msg)

    def _atualizar_pendentes(self) -> None:
        count = self.repo.contar_revisoes_pendentes()
        self.lbl_pendentes.setText(f"Revisao ({count})" if count else "Revisao (0)")
        if count > 0:
            self.lbl_pendentes.setStyleSheet(
                "QPushButton { background: #f44336; color: white; border-radius: 4px; padding: 4px 12px; font-weight: bold; } "
                "QPushButton:hover { background: #d32f2f; }"
            )
        else:
            self.lbl_pendentes.setStyleSheet(
                "QPushButton { background: #9e9e9e; color: white; border-radius: 4px; padding: 4px 12px; font-weight: bold; }"
            )

    def _on_voltar(self) -> None:
        self.voltar_clicked.emit()

    def _on_pausar(self) -> None:
        pass

    def _buscar_termo(self) -> None:
        if self.session.livro_id:
            TermSearchDialog(self.repo, self.session.livro_id, self).exec()

    def _contexto_camera(self) -> str:
        termo_i, termo_f = self.session.intervalo_termos_atual
        faixa = "?" if termo_i is None else (str(termo_i) if termo_i == termo_f else f"{termo_i}-{termo_f}")
        total = self.repo.get_total_imagens_livro(self.session.livro_id) if self.session.livro_id else 0
        return (
            f"Proxima captura: folha {self.session.ultima_folha or '?'} - "
            f"{(self.session.ultima_face or 'frente').capitalize()} - termos {faixa} | "
            f"{total} fotos registradas"
        )

    def _abrir_camera_automatica(self) -> None:
        if not self.session.livro_id:
            return
        indice = self.pipeline.settings.get("camera", "index", 0) if self.pipeline.settings else 0
        pasta = self.pipeline.acervo_root / f"livro_{self.session.livro_id}" / "capturas_camera"
        dlg = CameraCaptureDialog(pasta, self._contexto_camera, indice, self)
        dlg.foto_capturada.connect(self._processar_captura_camera)
        dlg.exec()

    def _processar_captura_camera(self, path: str) -> None:
        self.processar_nova_imagem(path)

    def _iniciar_proxima_captura(self) -> None:
        if self._captura_worker and self._captura_worker.isRunning():
            return
        if not self._captura_queue:
            return
        path = self._captura_queue.popleft()
        worker = CapturaPipelineWorker(self.pipeline, path)
        self._captura_worker = worker
        worker.concluido.connect(self._on_captura_concluida)
        worker.erro.connect(self._on_captura_erro)
        worker.finished.connect(self._on_captura_finalizada)
        worker.start()

    @Slot(dict)
    def _on_captura_concluida(self, result: dict) -> None:
        self._aplicar_resultado_captura(result)

    @Slot(str)
    def _on_captura_erro(self, mensagem: str) -> None:
        self._set_indicator(self.ind_imagem, "erro_grave", mensagem[:30])
        logger.error("Erro na captura em segundo plano: %s", mensagem)

    @Slot()
    def _on_captura_finalizada(self) -> None:
        if self._captura_worker:
            self._captura_worker.deleteLater()
        self._captura_worker = None
        self._iniciar_proxima_captura()

    def _on_finalizar(self) -> None:
        if self.session.livro_id:
            self.repo.atualizar_livro(self.session.livro_id, status="concluido")
        self.voltar_clicked.emit()
