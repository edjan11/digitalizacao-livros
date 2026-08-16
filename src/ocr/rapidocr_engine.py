"""Engine RapidOCR com ONNX Runtime (modelo embutido)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class RapidOCRError(Exception):
    pass


class RapidOCREngine:
    def __init__(self) -> None:
        self._engine = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            import os
            cpu_count = os.cpu_count() or 4
            self._engine = RapidOCR(params={
                "Global": {
                    "intra_op_num_threads": min(cpu_count, 4),
                    "inter_op_num_threads": 2,
                },
                "Det": {
                    "intra_op_num_threads": min(cpu_count, 4),
                    "inter_op_num_threads": 2,
                },
                "Cls": {
                    "intra_op_num_threads": min(cpu_count, 4),
                    "inter_op_num_threads": 2,
                },
                "Rec": {
                    "intra_op_num_threads": min(cpu_count, 4),
                    "inter_op_num_threads": 2,
                },
            })
            self._loaded = True
        except Exception as exc:
            raise RapidOCRError(f"RapidOCR indisponivel: {exc}") from exc

    def is_available(self) -> bool:
        try:
            self.load()
            return True
        except Exception:
            return False

    def read(self, image_path: Path) -> str:
        """OCR completo de uma imagem devolvendo texto bruto."""
        self.load()
        img = cv2.imread(str(image_path))
        if img is None:
            return ""
        result, _ = self._engine(img)
        if result is None:
            return ""
        lines = [item[1] for item in result]
        return "\n".join(lines)

    def read_array(self, image: "cv2.Mat | None") -> str:
        """OCR sobre buffer numpy."""
        if image is None:
            return ""
        self.load()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result, _ = self._engine(image)
        if result is None:
            return ""
        lines = [item[1] for item in result]
        return "\n".join(lines)

    def read_with_boxes(self, image: "cv2.Mat | None") -> list[tuple[list, str, float]]:
        """OCR retornando caixas + texto + confianca."""
        if image is None:
            return []
        self.load()
        result, _ = self._engine(image)
        if result is None:
            return []
        return [(item[0], item[1], item[2]) for item in result]