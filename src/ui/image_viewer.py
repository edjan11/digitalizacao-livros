from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QImageReader,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


class ImageViewer(QGraphicsView):
    """Visualizador de fotos em resolução nativa.

    A fotografia fica em um item imutável; moldura de registro e seleção são
    itens separados. Assim nenhum destaque é gravado no JPG nem reduz a
    qualidade enviada ao Qwen.
    """

    selection_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("#202124"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.Antialiasing
        )

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._highlight_item: QGraphicsRectItem | None = None
        self._selection_item: QGraphicsRectItem | None = None
        self._image_size = (0, 0)
        self._display_scale = 1.0
        self._zoom_percent: float | None = None
        self._selection_scene: QRectF | None = None
        self._selection_start: QPoint | None = None
        self._pan_start: QPoint | None = None
        self._selection_mode = True
        self._destaque_indice = 0
        self._destaque_total = 1
        self._destaque_texto = ""
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------
    # Carregamento sem modificar o arquivo original

    def set_image_path(
        self,
        path: str | Path,
        destaque_indice: int | None = None,
        total_registros: int = 1,
        texto_destaque: str = "",
    ) -> None:
        path = Path(path)
        if not path.is_file():
            self.clear()
            return
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self.clear()
            return
        self._set_qimage(image, destaque_indice, total_registros, texto_destaque)

    def set_image_array(
        self,
        image: np.ndarray,
        destaque_indice: int | None = None,
        total_registros: int = 1,
        texto_destaque: str = "",
        destaque_x_rel: tuple[float, float] | None = None,
    ) -> None:
        if image is None or image.size == 0:
            self.clear()
            return
        if image.ndim == 2:
            qimage = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        self._set_qimage(
            qimage, destaque_indice, total_registros, texto_destaque,
            destaque_x_rel,
        )

    def _set_qimage(
        self,
        image: QImage,
        destaque_indice: int | None,
        total_registros: int,
        texto_destaque: str,
        destaque_x_rel: tuple[float, float] | None = None,
    ) -> None:
        pixmap = QPixmap.fromImage(image)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._image_size = (pixmap.width(), pixmap.height())
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self._selection_scene = None
        self._selection_item = None
        self._highlight_item = None
        self._zoom_percent = None
        self._destaque_indice = 0 if destaque_indice is None else destaque_indice
        self._destaque_total = max(1, total_registros)
        self._destaque_texto = texto_destaque
        if destaque_indice is not None:
            self.set_highlight(
                destaque_indice, total_registros, texto_destaque,
                x_rel=destaque_x_rel,
            )
        self.fit_to_window()

    # ------------------------------------------------------------------
    # Moldura e seleção em coordenadas da imagem original

    def set_highlight(
        self,
        indice: int,
        total_registros: int = 1,
        texto: str = "",
        x_rel: tuple[float, float] | None = None,
    ) -> None:
        if not self._pixmap_item:
            return
        total = max(1, int(total_registros))
        indice = max(0, min(int(indice), total - 1))
        largura, altura = self._image_size
        if x_rel is None:
            # Moldura idêntica ao recorte usado pelo OCR/Qwen.
            from ..imaging.record_regions import bbox_registro
            bx1, by1, bx2, by2 = bbox_registro(indice, total)
            x1 = int(largura * bx1)
            y1 = int(altura * by1)
            x2 = int(largura * bx2)
            y2 = int(altura * by2)
        else:
            x1_rel = max(0.0, min(float(x_rel[0]), 1.0))
            x2_rel = max(x1_rel + 0.01, min(float(x_rel[1]), 1.0))
            x1 = int(largura * x1_rel)
            x2 = int(largura * x2_rel)
            margem_y = max(12, int(altura * 0.012))
            if total == 2:
                inicio_rel, fim_rel = ((0.0, 0.488), (0.488, 1.0))[indice]
                y1 = int(inicio_rel * altura)
                y2 = int(fim_rel * altura)
                if indice == 0:
                    y1 += margem_y
                else:
                    y2 -= margem_y
            else:
                bloco = altura / total
                y1 = int(indice * bloco) + margem_y
                y2 = int((indice + 1) * bloco) - margem_y

        if self._highlight_item is not None:
            self._scene.removeItem(self._highlight_item)
        self._highlight_item = self._scene.addRect(
            QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
            QPen(QColor("#ff5722"), max(3.0, largura * 0.0015)),
        )
        self._highlight_item.setBrush(Qt.BrushStyle.NoBrush)
        self._highlight_item.setZValue(2)
        self._highlight_item.setToolTip(texto)
        self._destaque_indice = indice
        self._destaque_total = total
        self._destaque_texto = texto

    def focus_highlight(self) -> None:
        if self._highlight_item is not None:
            self.centerOn(self._highlight_item.rect().center())

    def set_selection_mode(self, enabled: bool) -> None:
        self._selection_mode = bool(enabled)
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )

    def selection_mode(self) -> bool:
        return self._selection_mode

    def _set_selection_scene(self, rect: QRectF) -> None:
        rect = rect.normalized()
        image_rect = QRectF(0, 0, self._image_size[0], self._image_size[1])
        rect = rect.intersected(image_rect)
        if rect.width() < 2 or rect.height() < 2:
            return
        self._selection_scene = rect
        if self._selection_item is None:
            self._selection_item = self._scene.addRect(
                rect,
                QPen(QColor("#00bcd4"), max(3.0, self._image_size[0] * 0.0012), Qt.PenStyle.DashLine),
            )
            self._selection_item.setBrush(Qt.BrushStyle.NoBrush)
            self._selection_item.setZValue(3)
        else:
            self._selection_item.setRect(rect)
        self.selection_changed.emit()

    def _on_selection_finished(self, rect: QRect) -> None:
        """Compatibilidade com o teste/consumidor antigo que usa QRect de tela."""
        if not self._image_size or self._display_scale <= 0:
            return
        self._set_selection_scene(
            QRectF(
                rect.x() / self._display_scale,
                rect.y() / self._display_scale,
                rect.width() / self._display_scale,
                rect.height() / self._display_scale,
            )
        )

    def selected_relative_rect(self) -> tuple[float, float, float, float] | None:
        if self._selection_scene is None or not self._image_size[0] or not self._image_size[1]:
            return None
        rect = self._selection_scene
        return (
            max(0.0, rect.left() / self._image_size[0]),
            max(0.0, rect.top() / self._image_size[1]),
            min(1.0, rect.right() / self._image_size[0]),
            min(1.0, rect.bottom() / self._image_size[1]),
        )

    def clear_selection(self) -> None:
        self._selection_scene = None
        if self._selection_item is not None:
            self._scene.removeItem(self._selection_item)
            self._selection_item = None

    # ------------------------------------------------------------------
    # Zoom, ajuste e navegação

    def _set_transform_percent(self, percent: float, anchor: QPoint | None = None) -> None:
        if not self._pixmap_item:
            return
        percent = max(10.0, min(500.0, float(percent)))
        before = self.mapToScene(anchor) if anchor is not None else None
        self.resetTransform()
        self.scale(percent / 100.0, percent / 100.0)
        self._zoom_percent = percent
        self._display_scale = percent / 100.0
        if anchor is not None and before is not None:
            after = self.mapToScene(anchor)
            delta = after - before
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() + int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + int(delta.y())
            )

    def set_zoom_percent(self, percent: float) -> None:
        self._set_transform_percent(percent)

    def zoom_in(self) -> None:
        atual = self._zoom_percent or self._display_scale * 100.0
        self._set_transform_percent(atual * 1.25)

    def zoom_out(self) -> None:
        atual = self._zoom_percent or self._display_scale * 100.0
        self._set_transform_percent(atual / 1.25)

    def zoom_100(self) -> None:
        self._set_transform_percent(100.0)

    def fit_to_window(self) -> None:
        if not self._pixmap_item:
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_percent = None
        self._display_scale = self.transform().m11()

    def fit_to_width(self) -> None:
        if not self._pixmap_item or not self._image_size[0]:
            return
        largura = max(1, self.viewport().width() - 20)
        self._set_transform_percent(largura / self._image_size[0] * 100.0)

    def fit_to_page(self) -> None:
        self.fit_to_window()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom_percent is None:
            self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # Roda normaliza a navegação vertical/horizontal; Ctrl+roda é o
            # gesto explícito de zoom para não perder a posição na página.
            super().wheelEvent(event)
            return
        atual = self._zoom_percent or self._display_scale * 100.0
        fator = 1.25 if delta > 0 else 0.8
        self._set_transform_percent(atual * fator, event.position().toPoint())
        event.accept()

    # ------------------------------------------------------------------
    # Mouse: seleção no modo área; pan com botão do meio ou modo navegação

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._selection_mode:
            self._selection_start = event.position().toPoint()
            self.clear_selection()
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        atual = event.position().toPoint()
        if self._selection_start is not None:
            inicio = self.mapToScene(self._selection_start)
            fim = self.mapToScene(atual)
            self._set_selection_scene(QRectF(inicio, fim))
            event.accept()
            return
        if self._pan_start is not None:
            delta = atual - self._pan_start
            self._pan_start = atual
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._selection_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._selection_start = None
            event.accept()
            return
        if self._pan_start is not None and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._pan_start = None
            self.setCursor(
                Qt.CursorShape.CrossCursor
                if self._selection_mode
                else Qt.CursorShape.OpenHandCursor
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._highlight_item = None
        self._selection_item = None
        self._selection_scene = None
        self._image_size = (0, 0)
        self._zoom_percent = None
        self._display_scale = 1.0
