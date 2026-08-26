from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QGroupBox,
    QProgressBar, QMessageBox, QPlainTextEdit, QToolButton, QMenu,
)

from src.database.repository import Repository
from src.session.scan_session import ScanSession
from src.services.scan_pipeline import ScanPipeline
from src.services.telemetry import emitir, registrar_amostrador
from src.imaging.thumbnail import gerar_thumbnail
from src.imaging.quality import avaliar_qualidade
from src.imaging.enhance import melhorar_pagina
from .book_review_dialog import BookReviewDialog
from .term_search_dialog import TermSearchDialog
from .camera_capture_dialog import CameraCaptureDialog
from .image_viewer import ImageViewer
from .theme import (
    G_IR_ESQ, G_IR_DIR, G_ESPELHAR, G_CORTAR, G_OTIMIZAR, G_IMAGEM, G_FOCO,
    G_ENQUAD, G_DUP, G_OCR, G_SCANNER, TEXTO_SECUNDARIO, TEXTO_NEON, STATUS_OK,
    STATUS_ATENCAO, STATUS_ERRO, BTN_PRIMARIO, BTN_SECUNDARIO, BTN_ALERTA,
    status_visual,
)

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
        self._atualizar_breadcrumb()
        self._habilitar_transform(False)
        self._atualizar_pendentes()

    # ------------------------------------------------------------------
    # Construcao da interface (apenas visual)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # 2. Breadcrumb de hierarquia
        self.lbl_breadcrumb = QLabel()
        self.lbl_breadcrumb.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_breadcrumb.setFont(QFont("Segoe UI", 12))
        layout.addWidget(self.lbl_breadcrumb)

        # 3. Toolbar de acao rapida (canto superior direito)
        toolbar = QHBoxLayout()
        toolbar.addStretch()
        toolbar.addLayout(self._criar_grupo_acoes_imagem())
        toolbar.addSpacing(10)
        toolbar.addLayout(self._criar_grupo_estado())
        layout.addLayout(toolbar)

        # 4 + 5. Viewport central + sidebar de indicadores
        content = QHBoxLayout()
        content.setSpacing(12)
        content.addWidget(self._criar_viewport(), 1)
        content.addWidget(self._criar_indicadores(), 0)
        layout.addLayout(content)

        # Acoes de finalizacao / navegacao
        layout.addLayout(self._criar_acoes_fim())

        # 6. Barra de status + console
        layout.addLayout(self._criar_rodape())

    def _criar_grupo_acoes_imagem(self) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(6)
        btn_girar_esq = QPushButton(f"{G_IR_ESQ}  Girar Esq.")
        btn_girar_dir = QPushButton(f"{G_IR_DIR}  Girar Dir.")
        btn_espelhar = QToolButton()
        btn_espelhar.setText(f"{G_ESPELHAR}  Espelhar H/V")
        btn_espelhar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_esp = QMenu(btn_espelhar)
        menu_esp.addAction("Espelhar Horizontal", lambda: self._aplicar_transformacao(cv2.flip, 1, nome="espelhar_h"))
        menu_esp.addAction("Espelhar Vertical", lambda: self._aplicar_transformacao(cv2.flip, 0, nome="espelhar_v"))
        btn_espelhar.setMenu(menu_esp)
        btn_cortar = QPushButton(f"{G_CORTAR}  Cortar")
        btn_otimizar = QPushButton(f"{G_OTIMIZAR}  Otimizar")
        btn_girar_esq.clicked.connect(lambda: self._aplicar_transformacao(cv2.ROTATE_90_COUNTERCLOCKWISE, "girar_esq"))
        btn_girar_dir.clicked.connect(lambda: self._aplicar_transformacao(cv2.ROTATE_90_CLOCKWISE, "girar_dir"))
        btn_cortar.clicked.connect(self._cortar_foto)
        btn_otimizar.clicked.connect(self._otimizar_foto)
        for b in (btn_girar_esq, btn_girar_dir, btn_espelhar, btn_cortar, btn_otimizar):
            b.setStyleSheet(BTN_SECUNDARIO)
            b.setToolTip("Corrige a fotografia armazenada (persiste na imagem)")
        self._botoes_transform = (btn_girar_esq, btn_girar_dir, btn_espelhar, btn_cortar, btn_otimizar)
        for b in self._botoes_transform:
            h.addWidget(b)
        return h

    def _criar_grupo_estado(self) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(8)
        self.lbl_pendentes = QPushButton("Revisao (0)")
        self.lbl_pendentes.setStyleSheet(BTN_ALERTA)
        self.lbl_pendentes.clicked.connect(self.revisao_clicked.emit)
        btn_conferir = QPushButton("Conferir Livro")
        btn_conferir.setStyleSheet(BTN_PRIMARIO)
        btn_conferir.setToolTip("Resumo do livro: aprovadas, revisões, fotos a refazer e faltantes.")
        btn_conferir.clicked.connect(self._abrir_conferencia)
        btn_exportar = QPushButton("Exportar Termos")
        btn_exportar.setStyleSheet(BTN_SECUNDARIO)
        btn_exportar.clicked.connect(self._exportar_termos)
        btn_modo = QToolButton()
        btn_modo.setText("Modos de Captura  ▾")
        btn_modo.setStyleSheet(BTN_SECUNDARIO)
        btn_modo.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_modo = QMenu(btn_modo)
        menu_modo.addAction("Câmera automática", self._abrir_camera_automatica)
        menu_modo.addAction("Buscar termo", self._buscar_termo)
        btn_modo.setMenu(menu_modo)
        h.addWidget(self.lbl_pendentes)
        h.addWidget(btn_conferir)
        h.addWidget(btn_exportar)
        h.addWidget(btn_modo)
        return h

    def _criar_viewport(self) -> QFrame:
        self.foto_viewer = ImageViewer()
        self.foto_viewer.setMinimumSize(620, 430)
        self.foto_viewer.set_selection_mode(False)

        foto_panel = QFrame()
        foto_panel.setObjectName("panel")
        foto_layout = QVBoxLayout(foto_panel)
        foto_layout.setContentsMargins(10, 10, 10, 10)
        foto_layout.setSpacing(8)

        topo = QHBoxLayout()
        self.lbl_scanner = QLabel(f"{G_SCANNER}  SCANNER CONECTADO")
        self.lbl_scanner.setStyleSheet(f"color: {STATUS_OK}; font-weight: bold; font-size: 12px;")
        topo.addWidget(self.lbl_scanner)
        topo.addStretch()
        btn_fit = QPushButton("Ajustar à página")
        btn_fit.setToolTip("Mostrar a fotografia inteira")
        btn_fit.clicked.connect(self.foto_viewer.fit_to_page)
        btn_100 = QPushButton("100%")
        btn_100.clicked.connect(self.foto_viewer_zoom_100)
        for b in (btn_fit, btn_100):
            b.setStyleSheet(BTN_SECUNDARIO)
            topo.addWidget(b)
        foto_layout.addLayout(topo)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.foto_viewer, 0, 0)
        self._overlay = QLabel("Aguardando captura do scanner de mesa ou arquivo...")
        self._overlay.setStyleSheet(f"color: {TEXTO_NEON}; font-size: 14px;")
        self._overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._overlay, 0, 0)
        self._overlay.raise_()
        foto_layout.addLayout(grid, 1)

        self.lbl_foto_info = QLabel("Aguardando imagens...")
        self.lbl_foto_info.setStyleSheet(f"color: {TEXTO_SECUNDARIO}; font-size: 11px;")
        self.lbl_foto_info.setWordWrap(True)
        foto_layout.addWidget(self.lbl_foto_info)

        self.lbl_folha_face = QLabel("")
        self.lbl_folha_face.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_folha_face.setStyleSheet(f"color: {TEXTO_NEON};")
        self.lbl_termo = QLabel("")
        self.lbl_termo.setStyleSheet(f"color: {TEXTO_NEON};")
        info = QHBoxLayout()
        info.addWidget(self.lbl_folha_face)
        info.addStretch()
        info.addWidget(self.lbl_termo)
        foto_layout.addLayout(info)
        return foto_panel

    def _criar_indicadores(self) -> QGroupBox:
        box = QGroupBox("Indicadores")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)
        self._indicadores = {}
        for chave, nome, glyph in (
            ("imagem", "Imagem Qualidade", G_IMAGEM),
            ("foco", "Foco", G_FOCO),
            ("enquadramento", "Enquadramento", G_ENQUAD),
            ("duplicidade", "Duplicidade", G_DUP),
            ("ocr", "OCR", G_OCR),
        ):
            row = QHBoxLayout()
            icone = QLabel(glyph)
            icone.setStyleSheet(f"color: {TEXTO_NEON}; font-size: 13px;")
            row.addWidget(icone)
            nome_lbl = QLabel(nome + ":")
            nome_lbl.setStyleSheet(f"color: {TEXTO_NEON};")
            row.addWidget(nome_lbl)
            row.addStretch()
            status = QLabel("•")
            status.setStyleSheet(f"color: {TEXTO_SECUNDARIO}; font-weight: bold; font-size: 13px;")
            row.addWidget(status)
            layout.addLayout(row)
            self._indicadores[chave] = status
        layout.addStretch()
        box.setMinimumWidth(240)
        box.setMaximumWidth(300)
        return box

    def _criar_acoes_fim(self) -> QHBoxLayout:
        h = QHBoxLayout()
        btn_voltar = QPushButton("Trocar Livro")
        btn_voltar.setStyleSheet(BTN_SECUNDARIO)
        btn_voltar.clicked.connect(self._on_voltar)
        h.addWidget(btn_voltar)
        btn_pausar = QPushButton("Pausar")
        btn_pausar.setStyleSheet(BTN_SECUNDARIO)
        btn_pausar.clicked.connect(self._on_pausar)
        h.addWidget(btn_pausar)
        h.addStretch()
        self.btn_finalizar = QPushButton("Finalizar Livro")
        self.btn_finalizar.setStyleSheet(BTN_PRIMARIO)
        self.btn_finalizar.clicked.connect(self._on_finalizar)
        h.addWidget(self.btn_finalizar)
        return h

    def _criar_rodape(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumHeight(72)
        self._console.setStyleSheet(
            f"color: {TEXTO_SECUNDARIO}; font-size: 10px; border: 1px solid #343B48; border-radius: 4px;"
        )
        h.addWidget(self._console, 1)
        self.lbl_status = QLabel("Digitalização automática de mesa ativada.")
        self.lbl_status.setStyleSheet(f"color: {TEXTO_NEON}; font-size: 10px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(self.lbl_status, 0)
        return h

    # ------------------------------------------------------------------
    # Indicadores / status (apenas visual)

    def _set_indicator(self, chave: str, status: str, value: str = "") -> None:
        label = self._indicadores.get(chave)
        if label is None:
            return
        glyph, cor = status_visual(status)
        texto = glyph
        if value:
            texto = f"{glyph}  {value}"
        label.setText(texto)
        label.setStyleSheet(f"color: {cor}; font-weight: bold; font-size: 13px;")
        label.setToolTip((value or status).upper())

    def _ind_tooltip(self, chave: str, texto: str) -> None:
        label = self._indicadores.get(chave)
        if label is not None:
            label.setToolTip(texto)

    def _atualizar_breadcrumb(self) -> None:
        oficio = self.repo.get_oficio(self.session.oficio_id) if self.session.oficio_id else None
        tipo = self.repo.get_tipo(self.session.tipo_id) if self.session.tipo_id else None
        livro = (self.session.livro or {}).get("codigo") or ""
        o = (oficio or {}).get("nome") or f"{self.session.oficio_id}º Ofício"
        t = (tipo or {}).get("nome") or "Tipo"
        self.lbl_breadcrumb.setText(
            f"<span style='color: {TEXTO_SECUNDARIO}'>{o}</span> &nbsp;&gt;&nbsp; "
            f"<span style='color: {TEXTO_NEON}'>{t}</span> &nbsp;&gt;&nbsp; "
            f"<span style='color: {TEXTO_NEON}; font-weight: bold;'>{livro}</span>"
        )

    def set_scanner_status(self, conectado: bool) -> None:
        if conectado:
            self.lbl_scanner.setText(f"{G_SCANNER}  SCANNER CONECTADO")
            self.lbl_scanner.setStyleSheet(f"color: {STATUS_OK}; font-weight: bold; font-size: 12px;")
            self.log("Scanner de Mesa OK.")
        else:
            self.lbl_scanner.setText(f"{G_SCANNER}  AGUARDANDO SCANNER")
            self.lbl_scanner.setStyleSheet(f"color: {TEXTO_SECUNDARIO}; font-weight: bold; font-size: 12px;")

    def log(self, mensagem: str) -> None:
        self._console.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] - {mensagem}")

    def _set_overlay_visivel(self, visivel: bool) -> None:
        self._overlay.setVisible(visivel)

    def _otimizar_foto(self) -> None:
        p = self._caminho_edicao()
        if p is None:
            return
        image = cv2.imread(str(p))
        if image is None:
            return
        melhorada, info = melhorar_pagina(image, ativo=True)
        self._salvar_transformada(p, melhorada, "otimizar: auto" if info.get("aplicado") else "otimizar: original")
        self.log("Foto otimizada (recorte/perspectiva/clareamento).")

    def _exportar_termos(self) -> None:
        QMessageBox.information(
            self, "Exportar Termos",
            "A exportação de termos será disponibilizada em uma próxima versão."
        )

    # ------------------------------------------------------------------
    # Fluxo de captura (logica preservada)

    def foto_viewer_zoom_100(self) -> None:
        self.foto_viewer.zoom_100()

    def processar_nova_imagem(self, image_path: str) -> None:
        self._captura_queue.append(str(image_path))
        self._set_indicator(
            "imagem", "aviso",
            f"fila: {len(self._captura_queue) + (1 if self._captura_worker else 0)}",
        )
        self._iniciar_proxima_captura()

    def _aplicar_resultado_captura(self, result: dict) -> None:
        self._ultimo_resultado = result
        self._ultima_imagem_id = result.get("imagem_id")
        self._habilitar_transform(True)
        self._atualizar_ui_imediatamente(result)
        if result.get("aguarda_confirmacao_duplicidade") and self._ultima_imagem_id:
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
            self._set_indicator("duplicidade", "revisar", "CONFIRMADA")
        else:
            self._set_indicator("duplicidade", "ok", "NAO E DUPLICATA")
        self._atualizar_pendentes()

    def _atualizar_ui_imediatamente(self, result: dict) -> None:
        self._mostrar_foto_resultado(result)
        folha = result.get("folha")
        face = result.get("face", "")
        if folha:
            self.log(f"Pagina {folha} ({face.capitalize()}) capturada.")
        q = result.get("qualidade", {})
        if result.get("nao_registro"):
            self._set_indicator("imagem", "revisar", "SEM TERMO / CONTAGEM PARADA")
            self._set_indicator("ocr", "aviso", "NAO EXECUTADO")
            self.lbl_folha_face.setText("Documento preservado — não é face de registro")
            self.lbl_termo.setText("Nenhum termo atribuído; contador não avançou")
            return
        if q.get("repetir_captura"):
            motivos = ", ".join(q.get("motivos_refazer", []))
            self._set_indicator("imagem", "revisar", "REFAZER DEPOIS")
            self._ind_tooltip("imagem", motivos)
        else:
            self._set_indicator("imagem", q.get("status_geral", "ok"))
        self._set_indicator("foco", q.get("foco_status", "ok"))
        self._set_indicator("enquadramento", q.get("enquadramento_status", "ok"))
        dup = result.get("duplicidade", {})
        self._set_indicator("duplicidade", "revisar" if dup.get("status") != "unico" else "ok")
        self.lbl_folha_face.setText(
            f"Folha: {folha} - {face.capitalize()}" if folha else "Aguardando imagens..."
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
        self._set_overlay_visivel(False)
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
        folha = result.get("folha")
        if result.get("imagem_id") == self._ultima_imagem_id:
            if termo_inicial is not None:
                faixa = str(termo_inicial) if termo_inicial == termo_final else f"{termo_inicial}-{termo_final}"
                self.lbl_termo.setText(f"Termos: {faixa} ({status})")
            else:
                self.lbl_termo.setText(f"Termo OCR: {termo or '?'} ({status})")
            if folha:
                self.lbl_folha_face.setText(f"Folha: {folha} - {self.session.ultima_face.capitalize()}")
            self._set_indicator("ocr", status)
        motor = result.get("motor", "")
        tempos = result.get("tempos_ms", {})
        self._ind_tooltip("ocr", f"Motor: {motor}\nTempos: {tempos}")
        self.log(f"OCR concluido — termo {termo_inicial if termo_inicial is not None else termo}")
        self._atualizar_pendentes()

    @Slot(str)
    def _on_ocr_erro(self, msg: str) -> None:
        self._set_indicator("ocr", "erro_grave", msg[:30])
        self.log(f"Erro OCR: {msg}")
        logger.error("Erro OCR: %s", msg)

    def _atualizar_pendentes(self) -> None:
        count = self.repo.contar_revisoes_pendentes()
        self.lbl_pendentes.setText(f"Revisao ({count})" if count else "Revisao (0)")
        if count > 0:
            self.lbl_pendentes.setStyleSheet(BTN_ALERTA)
        else:
            self.lbl_pendentes.setStyleSheet(BTN_PRIMARIO)

    # ------------------------------------------------------------------
    # Ferramentas de correcao da foto (giro, espelho, corte, otimizar)

    def _habilitar_transform(self, estado: bool) -> None:
        for b in self._botoes_transform:
            b.setEnabled(estado)

    def _caminho_edicao(self) -> Path | None:
        if not self._ultima_imagem_id:
            return None
        img = self.repo.get_imagem(self._ultima_imagem_id)
        if not img:
            return None
        p = Path(img.get("caminho_armazenamento") or img.get("caminho_original"))
        return p if p.is_file() else None

    def _aplicar_transformacao(self, op, *args, nome: str = "transformar") -> None:
        p = self._caminho_edicao()
        if p is None:
            return
        image = cv2.imread(str(p))
        if image is None:
            return
        if op is cv2.flip:
            image = cv2.flip(image, *args)
        else:
            image = cv2.rotate(image, op)
        self._salvar_transformada(p, image, f"correcao: {nome}")

    def _cortar_foto(self) -> None:
        if not self._ultima_imagem_id:
            return
        rect = self.foto_viewer.selected_relative_rect()
        if rect is None:
            QMessageBox.information(
                self, "Cortar", "Selecione primeiro uma area na foto com o mouse."
            )
            return
        p = self._caminho_edicao()
        if p is None:
            return
        image = cv2.imread(str(p))
        if image is None:
            return
        h, w = image.shape[:2]
        x1 = max(0, min(int(rect[0] * w), w - 1))
        x2 = max(x1 + 1, min(int(rect[2] * w), w))
        y1 = max(0, min(int(rect[1] * h), h - 1))
        y2 = max(y1 + 1, min(int(rect[3] * h), h))
        crop = image[y1:y2, x1:x2]
        self._salvar_transformada(p, crop, "corte de area")

    def _salvar_transformada(self, p: Path, image: np.ndarray, motivo: str) -> None:
        cv2.imwrite(str(p), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        self._regenerar_thumb(p)
        q = self._reavaliar_pos_transformacao(image)
        self.foto_viewer.set_image_path(p)
        self.foto_viewer.fit_to_page()
        self._set_indicator("imagem", q.get("status_geral", "ok"), motivo)
        self._set_indicator("foco", q.get("foco_status", "ok"))
        self._set_indicator("enquadramento", q.get("enquadramento_status", "ok"))
        self._atualizar_pendentes()
        logger.info("Foto corrigida e salva em %s (%s)", p, motivo)

    def _regenerar_thumb(self, p: Path) -> None:
        img = self.repo.get_imagem(self._ultima_imagem_id)
        thumb_path = Path(
            img.get("caminho_thumb")
            or (Path(p).parent.parent / "thumbs" / f"{Path(p).stem}_thumb.jpg")
        )
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        orig = cv2.imread(str(p))
        if orig is not None:
            cv2.imwrite(str(thumb_path), gerar_thumbnail(orig), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        self.repo.atualizar_imagem(
            self._ultima_imagem_id, rotacao_visualizacao=0, caminho_thumb=str(thumb_path)
        )

    def _reavaliar_pos_transformacao(self, image: np.ndarray) -> dict:
        """Reavalia a qualidade apos corrigir a foto e limpa a pendencia de
        refotografia quando a orientacao era o unico motivo da revisao."""
        partes = Path(self._caminho_edicao()).parts if self._caminho_edicao() else ()
        q = avaliar_qualidade(image, exigir_margens="capturas_camera" in partes)
        self.repo.atualizar_imagem(
            self._ultima_imagem_id,
            qualidade_foco=q.get("foco_valor"),
            qualidade_exposicao=q.get("exposicao_valor"),
            qualidade_enquadramento=q.get("enquadramento_status"),
            qualidade_orientacao=0,
            qualidade_status=q.get("status_geral"),
            qualidade_oclusao=q.get("oclusao_valor"),
            qualidade_motivos="; ".join(q.get("motivos_refazer", [])),
        )
        if not q.get("repetir_captura"):
            self.repo.atualizar_imagem(self._ultima_imagem_id, precisa_revisao=0)
            self.repo.resolver_revisoes_tipo(self._ultima_imagem_id, "refazer_captura")
        try:
            cl = self.pipeline._classificar_documento(image)
            self.repo.atualizar_imagem(self._ultima_imagem_id, tipo_documento=cl.get("tipo"))
            if cl.get("tipo") == "registro":
                self.repo.resolver_revisoes_tipo(self._ultima_imagem_id, "classificar_documento")
        except Exception:
            logger.exception("Falha ao reclassificar apos transformacao")
        return q

    def _abrir_conferencia(self) -> None:
        livro_id = self.session.livro_id
        if not livro_id:
            QMessageBox.information(
                self, "Conferir livro", "Selecione um livro antes de conferir."
            )
            return
        dlg = BookReviewDialog(self.repo, self.pipeline, int(livro_id), self)
        dlg.exec()
        self._atualizar_pendentes()

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
        self._set_indicator("imagem", "erro_grave", mensagem[:30])
        self.log(f"Erro na captura: {mensagem}")
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
