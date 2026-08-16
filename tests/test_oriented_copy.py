from __future__ import annotations

import hashlib

import cv2
import numpy as np

from src.imaging.oriented_copy import aplicar_rotacao, materializar_copia_orientada


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rotacao_e_copia_externa_nao_alteram_original(tmp_path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :10] = (0, 0, 255)
    image[:, 10:20] = (0, 255, 0)
    image[:, 20:] = (255, 0, 0)
    origem = tmp_path / "foto com acento.jpg"
    assert cv2.imwrite(str(origem), image)
    antes = _hash(origem)

    derivada = materializar_copia_orientada(origem, 180, cache_dir=tmp_path / "cache")
    assert derivada.is_file()
    assert derivada != origem
    assert _hash(origem) == antes
    orientada = cv2.imread(str(derivada))
    assert orientada is not None
    assert orientada.shape[:2] == image.shape[:2]
    assert np.mean(orientada[:, :10, 0]) > 150
    assert np.mean(orientada[:, -10:, 2]) > 150


def test_rotacao_90_troca_dimensoes_sem_mutar_array():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    rotated = aplicar_rotacao(image, 90)
    assert rotated.shape[:2] == (30, 20)
    assert image.shape[:2] == (20, 30)
