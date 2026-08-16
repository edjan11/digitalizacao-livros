from __future__ import annotations

import cv2
import numpy as np

from src.imaging.page_orientation import OrientationDetector, OrientationResult
from src.services.orientation_processing import decidir_orientacao_coorte


def _page() -> np.ndarray:
    image = np.full((800, 520, 3), 245, dtype=np.uint8)
    for y in range(80, 760, 35):
        cv2.line(image, (45, y), (475, y), (80, 80, 80), 2)
    cv2.line(image, (80, 30), (80, 770), (50, 50, 50), 3)
    cv2.line(image, (390, 30), (390, 770), (50, 50, 50), 3)
    image[10:45, 10:45] = (0, 0, 255)
    return image


def _markers(image: np.ndarray) -> tuple[str, float]:
    # Simula o RapidOCR dos rotulos impressos: a marca vermelha representa o
    # cabecalho legivel somente quando a pagina esta em pe.
    corner = image[: image.shape[0] // 4, : image.shape[1] // 4]
    red = (corner[:, :, 2] > 220) & (corner[:, :, 1] < 80)
    if int(red.sum()) > 500:
        return "Numero Em de mil novecentos que recebeu o nome de", 0.98
    return "Notas Averbacoes e Retificacoes", 0.45


def test_orientacao_corrige_as_quatro_rotacoes():
    detector = OrientationDetector(marker_reader=_markers)
    original = _page()
    entradas = {
        0: original,
        90: cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(original, cv2.ROTATE_180),
        270: cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    esperadas = {0: 0, 90: 270, 180: 180, 270: 90}
    for entrada, image in entradas.items():
        result = detector.detect(image)
        assert result.rotation == esperadas[entrada]
        assert result.confidence >= 0.85
        assert result.auto_apply is True


def test_orientacao_inconclusiva_bloqueia_ocr():
    detector = OrientationDetector(marker_reader=lambda _image: ("", 0.0))
    result = detector.detect(np.full((400, 300, 3), 240, dtype=np.uint8))
    assert result.auto_apply is False
    assert result.needs_review is True
    assert result.confidence < 0.85


def _result(rotation: int, confidence: float) -> OrientationResult:
    return OrientationResult(
        rotation=rotation, confidence=confidence, method="teste", reason="teste",
        auto_apply=confidence >= 0.85, needs_review=confidence < 0.85, scores={},
    )


def test_coorte_forte_supera_um_falso_180_de_baixa_confianca():
    decision = decidir_orientacao_coorte([
        _result(0, 0.88), _result(0, 0.79), _result(0, 0.88),
        _result(180, 0.27), _result(0, 0.61),
    ])
    assert decision.approved
    assert decision.rotation == 0
    assert decision.confidence >= 0.85


def test_coorte_nao_aprova_conflito_forte():
    decision = decidir_orientacao_coorte([
        _result(0, 0.91), _result(0, 0.90), _result(180, 0.92),
    ])
    assert not decision.approved
