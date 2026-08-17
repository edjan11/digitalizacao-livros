from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Callable

import cv2

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..capture.auto_capture import AutoCaptureController, FrameAnalysis
from ..services.telemetry import emitir


class CameraCaptureDialog(QDialog):
    foto_capturada = Signal(str)

    def __init__(
        self,
        capture_dir: Path,
        context_provider: Callable[[], str],
        camera_index: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.capture_dir = capture_dir
        self.context_provider = context_provider
        self._cap = None
        self._frame = None
        self._controller = AutoCaptureController()
        self._ultimo_status: str | None = None
        self._estado_inicio = time.monotonic()
        self._janela_inicio = time.monotonic()
        self._janela_frames = 0
        self._janela_detector_ms = 0.0
        self._janela_save_ms = 0.0
        self._janela_jitter_ms = 0.0
        self._ultimo_tick = time.monotonic()
        self.setWindowTitle("Captura automatica por camera")
        self.setMinimumSize(1050, 760)
        self._init_ui(camera_index)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._atualizar_frame)
        self.finished.connect(lambda _result: self._parar_camera())

    def _init_ui(self, camera_index: int) -> None:
        layout = QVBoxLayout(self)
        self.contexto = QLabel(self.context_provider())
        self.contexto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.contexto.setStyleSheet("font-size: 16px; font-weight: bold; color: #1565c0;")
        layout.addWidget(self.contexto)

        self.preview = QLabel("Camera parada")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(960, 590)
        self.preview.setStyleSheet("background: #161616; color: white;")
        layout.addWidget(self.preview, 1)

        self.status = QLabel("Posicione a pagina dentro da guia")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.status)

        controles = QHBoxLayout()
        controles.addWidget(QLabel("Camera:"))
        self.indice = QSpinBox()
        self.indice.setRange(0, 9)
        self.indice.setValue(camera_index)
        controles.addWidget(self.indice)
        self.automatica = QCheckBox("Captura automatica")
        self.automatica.setChecked(True)
        controles.addWidget(self.automatica)
        self.btn_iniciar = QPushButton("Iniciar camera")
        self.btn_iniciar.clicked.connect(self._alternar_camera)
        controles.addWidget(self.btn_iniciar)
        self.btn_manual = QPushButton("Capturar agora")
        self.btn_manual.clicked.connect(self._capturar_manual)
        self.btn_manual.setEnabled(False)
        controles.addWidget(self.btn_manual)
        controles.addStretch()
        fechar = QPushButton("Fechar")
        fechar.clicked.connect(self.accept)
        controles.addWidget(fechar)
        layout.addLayout(controles)

    def _alternar_camera(self) -> None:
        if self._cap is not None:
            self._parar_camera()
            return
        indice = self.indice.value()
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        cap = cv2.VideoCapture(indice, backend)
        if not cap.isOpened():
            cap.release()
            QMessageBox.warning(
                self,
                "Camera nao encontrada",
                f"Nao foi possivel abrir a camera {indice}.\n"
                "A modalidade por pasta/CZUR continua disponivel.",
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._cap = cap
        self._controller.reset()
        self.indice.setEnabled(False)
        self.btn_iniciar.setText("Parar camera")
        self.btn_manual.setEnabled(True)
        self._timer.start(50)

    def _parar_camera(self) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._frame = None
        self.indice.setEnabled(True)
        self.btn_iniciar.setText("Iniciar camera")
        self.btn_manual.setEnabled(False)
        self.preview.setText("Camera parada")

    def _atualizar_frame(self) -> None:
        if self._cap is None:
            return
        agora = time.monotonic()
        tick = agora - self._ultimo_tick
        self._ultimo_tick = agora
        self._janela_jitter_ms = max(self._janela_jitter_ms, tick * 1000)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.status.setText("Falha ao ler a camera")
            return
        self._frame = frame
        inicio_detector = time.perf_counter()
        analise = self._controller.analisar(frame)
        self._janela_detector_ms = max(
            self._janela_detector_ms, (time.perf_counter() - inicio_detector) * 1000
        )
        self._janela_frames += 1
        if self._ultimo_status != analise.status:
            emitir("capture.state", de=self._ultimo_status or "inicio",
                   para=analise.status,
                   duracao_estado_ms=round((agora - self._estado_inicio) * 1000))
            self._ultimo_status = analise.status
            self._estado_inicio = agora
        if agora - self._janela_inicio >= 1.0:
            duracao = agora - self._janela_inicio
            emitir("capture.sample", fps_preview=round(self._janela_frames / duracao, 1),
                   fps_analisado=round(self._janela_frames / duracao, 1),
                   detector_ms_max=round(self._janela_detector_ms, 1),
                   save_ms_max=round(self._janela_save_ms, 1),
                   ui_jitter_ms_max=round(self._janela_jitter_ms, 1),
                   contagem=self._janela_frames)
            self._janela_inicio = agora
            self._janela_frames = 0
            self._janela_detector_ms = 0.0
            self._janela_save_ms = 0.0
            self._janela_jitter_ms = 0.0
        if analise.capturar:
            if self.automatica.isChecked():
                self._salvar_captura(frame)
            else:
                self._controller.reset()
                analise.capturar = False
                analise.status = "Pronta para captura manual"
        self._mostrar_preview(frame, analise)
        self.contexto.setText(self.context_provider())

    def _mostrar_preview(self, frame, analise: FrameAnalysis) -> None:
        exibida = frame.copy()
        h, w = exibida.shape[:2]
        x1, y1, x2, y2 = int(w * 0.06), int(h * 0.04), int(w * 0.94), int(h * 0.96)
        # Apenas linhas finas sobre a previa; nada cobre o texto da pagina.
        cv2.rectangle(
            exibida, (x1, y1), (x2, y2), (210, 150, 30),
            max(1, int(w * 0.0015)),
        )
        if analise.pagina_contorno is not None:
            pontos = analise.pagina_contorno.copy()
            pontos[:, 0] *= w
            pontos[:, 1] *= h
            cor_pagina = (0, 190, 0) if analise.enquadrada else (0, 80, 230)
            cv2.polylines(
                exibida,
                [pontos.astype("int32").reshape(-1, 1, 2)],
                True,
                cor_pagina,
                max(2, int(w * 0.002)),
            )
        texto = analise.status
        if analise.contagem is not None and analise.contagem > 0:
            texto = f"{texto}: {analise.contagem:.1f}s"
        self.status.setText(
            f"{texto} | foco {analise.foco:.0f} | movimento {analise.movimento:.1f}"
        )
        rgb = cv2.cvtColor(exibida, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def _capturar_manual(self) -> None:
        if self._frame is not None:
            self._controller.marcar_capturada(self._frame)
            self._salvar_captura(self._frame)

    def _salvar_captura(self, frame) -> None:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        nome = datetime.now().strftime("CAM_%Y%m%d_%H%M%S_%f.jpg")
        path = self.capture_dir / nome
        inicio = time.perf_counter()
        ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        duracao_ms = (time.perf_counter() - inicio) * 1000
        self._janela_save_ms = max(self._janela_save_ms, duracao_ms)
        emitir("capture.taken", duration_ms=round(duracao_ms, 1))
        if ok:
            self.foto_capturada.emit(str(path))

    def closeEvent(self, event) -> None:
        self._parar_camera()
        event.accept()
