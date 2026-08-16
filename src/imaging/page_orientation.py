from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import cv2
import numpy as np
import unicodedata


MarkerReader = Callable[[np.ndarray], tuple[str, float]]


@dataclass(frozen=True)
class OrientationResult:
    rotation: int
    confidence: float
    method: str
    reason: str
    auto_apply: bool
    needs_review: bool
    scores: dict[int, float]


def rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    rotation = int(rotation) % 360
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


class OrientationDetector:
    """Orienta formularios por estrutura e rotulos impressos.

    As quatro rotacoes sao avaliadas. A geometria escolhe o eixo provavel e o
    leitor de rotulos diferencia topo e base. OSD pode ser fornecido como
    desempate, mas nunca decide sozinho.
    """

    MARKERS = (
        "mil novecentos", "cartorio compareceu", "recebeu o nome",
        "nasceu uma crianca", "filh", "avos paternos",
    )

    def __init__(
        self,
        marker_reader: MarkerReader | None = None,
        osd_reader: Callable[[np.ndarray], tuple[int, float]] | None = None,
    ) -> None:
        self._marker_reader = marker_reader
        self._osd_reader = osd_reader
        self._rapid = None

    @staticmethod
    def _structure_score(image: np.ndarray) -> float:
        h, w = image.shape[:2]
        portrait = min(1.0, max(0.0, (h / max(w, 1) - 1.0) / 0.35))
        small_w = min(700, w)
        scale = small_w / max(w, 1)
        small = cv2.resize(image, (small_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=max(50, int(small_w * 0.28)), maxLineGap=25,
        )
        horizontal = vertical = 0.0
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
                length = float(np.hypot(dx, dy))
                if dx >= max(1, dy * 5):
                    horizontal += length
                elif dy >= max(1, dx * 5):
                    vertical += length
        line_score = min(1.0, horizontal / max(1.0, small_w * 12.0))
        # Formularios corretos sao retrato e possuem muito mais pauta
        # horizontal que linhas verticais.
        dominance = horizontal / max(horizontal + vertical, 1.0)
        return 0.45 * portrait + 0.35 * line_score + 0.20 * dominance

    def _default_marker_reader(self, image: np.ndarray) -> tuple[str, float]:
        if self._rapid is None:
            from ..ocr.rapidocr_engine import RapidOCREngine
            self._rapid = RapidOCREngine()
        h, w = image.shape[:2]
        scale = min(1.0, 1100 / max(w, 1))
        top = image[: max(1, int(h * 0.62))]
        if scale < 1.0:
            top = cv2.resize(
                top, (max(1, int(w * scale)), max(1, int(top.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        boxes = self._rapid.read_with_boxes(top)
        if not boxes:
            return "", 0.0
        text = " ".join(str(item[1]) for item in boxes)
        confidence = float(sum(float(item[2]) for item in boxes) / len(boxes))
        return text, confidence

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = "".join(
            character for character in unicodedata.normalize("NFD", str(value))
            if unicodedata.category(character) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _positioned_marker_score(self, image: np.ndarray) -> tuple[float, str]:
        """Use both label recognition and its printed position.

        RapidOCR can recognize an upside-down form after internally rotating a
        text line.  The printed labels then appear on the *right* side of the
        physical page, however.  In A-series birth books the labels that anchor
        the record (cartorio, name, filiation and grandparents) are on the left.
        This positional evidence is what safely separates 0 from 180 degrees.
        """
        if self._rapid is None:
            from ..ocr.rapidocr_engine import RapidOCREngine
            self._rapid = RapidOCREngine()
        h, w = image.shape[:2]
        scale = min(1.0, 1100 / max(w, 1))
        top = image[: max(1, int(h * 0.62))]
        if scale < 1.0:
            top = cv2.resize(
                top, (max(1, int(w * scale)), max(1, int(top.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        boxes = self._rapid.read_with_boxes(top)
        if not boxes:
            return 0.0, "nenhum rotulo localizado"

        anchors = (
            ("cartorio", 1.0), ("recebeu o nome", 1.2), ("nasceu uma", 0.8),
            ("filh", 0.65), ("avos paternos", 0.8), ("mil novecentos", 0.55),
        )
        evidence = total_weight = confidence_sum = 0.0
        found = 0
        for box, text, confidence in boxes:
            normalized = self._normalize_text(text)
            matched_weight = max(
                (weight for marker, weight in anchors if marker in normalized),
                default=0.0,
            )
            if not matched_weight:
                continue
            xs = [float(point[0]) for point in box]
            center_x = (min(xs) + max(xs)) / 2.0 / max(top.shape[1], 1)
            # Expected record labels occupy the left 30%.  Anything beyond the
            # middle is mirrored evidence and therefore penalizes the score.
            position = 1.0 if center_x <= 0.32 else (0.35 if center_x <= 0.50 else 0.0)
            evidence += matched_weight * position * float(confidence)
            total_weight += matched_weight
            confidence_sum += float(confidence)
            found += 1
        if not found:
            return 0.0, "nenhum rotulo-ancora localizado"
        coverage = min(1.0, evidence / 2.8)
        mean_confidence = confidence_sum / found
        score = coverage * max(0.0, min(1.0, mean_confidence))
        return score, f"{found} rotulos posicionais; OCR={mean_confidence:.2f}"

    def _marker_score(self, image: np.ndarray) -> tuple[float, str]:
        if self._marker_reader is None:
            try:
                return self._positioned_marker_score(image)
            except Exception as exc:
                return 0.0, f"RapidOCR indisponivel: {exc}"
        reader = self._marker_reader
        try:
            text, ocr_conf = reader(image)
        except Exception as exc:
            return 0.0, f"RapidOCR indisponivel: {exc}"
        normalized = self._normalize_text(text)
        found = sum(marker in normalized for marker in self.MARKERS)
        # "Notas/Averbacoes" aparece no lado direito e, isoladamente, nao
        # prova que o cabecalho esteja para cima.
        # Injected readers are used by deterministic tests/camera adapters and
        # normally return only the strongest header phrases.  Two independent
        # anchors are sufficient; the production reader additionally verifies
        # their physical x-position above.
        score = min(1.0, found / 2.0) * max(0.0, min(1.0, float(ocr_conf)))
        return score, f"{found} rotulos do formulario; OCR={ocr_conf:.2f}"

    def detect(self, image: np.ndarray) -> OrientationResult:
        if image is None or image.size == 0:
            return OrientationResult(0, 0.0, "sem_evidencia", "imagem vazia", False, True, {})
        structure = {
            rotation: self._structure_score(rotate_image(image, rotation))
            for rotation in (0, 90, 180, 270)
        }
        # OCR e a etapa cara: roda nas duas orientacoes que compartilham o
        # melhor eixo, o suficiente para diferenciar topo/base.
        candidates = sorted(structure, key=structure.get, reverse=True)[:2]
        scores = {rotation: structure[rotation] * 0.55 for rotation in structure}
        reasons: dict[int, str] = {}
        for rotation in candidates:
            marker, reason = self._marker_score(rotate_image(image, rotation))
            scores[rotation] += marker * 0.45
            reasons[rotation] = reason
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_rotation, best_score = ordered[0]
        gap = max(0.0, best_score - ordered[1][1])
        marker_best = max(0.0, (best_score - structure[best_rotation] * 0.55) / 0.45)
        confidence = min(0.99, 0.50 * marker_best + 0.25 * structure[best_rotation] + 0.50 * gap)

        method = "linhas+rapidocr"
        if confidence < 0.85 and gap < 0.10 and self._osd_reader is not None:
            try:
                osd_rotation, osd_confidence = self._osd_reader(image)
                if osd_rotation == best_rotation and osd_confidence >= 0.75:
                    confidence = min(0.84, confidence + 0.10)
                    method += "+osd_desempate"
            except Exception:
                pass
        auto = confidence >= 0.85
        reason = (
            f"rotacao {best_rotation}; estrutura={structure[best_rotation]:.2f}; "
            f"margem={gap:.2f}; {reasons.get(best_rotation, 'sem rotulos')}"
        )
        return OrientationResult(
            rotation=best_rotation,
            confidence=confidence,
            method=method,
            reason=reason,
            auto_apply=auto,
            needs_review=not auto,
            scores={key: round(value, 5) for key, value in scores.items()},
        )
