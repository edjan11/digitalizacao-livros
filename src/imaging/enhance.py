"""Melhoria de pagina estilo scanner de documento (Adobe Scan).

Recorta a pagina pelo contorno, corrige a perspectiva e uniformiza o fundo
(remocao de sombra por divisao). Conservador: sem quadrilatero confiavel,
devolve a foto original intacta — nunca inventa conteudo.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

AREA_MINIMA_PAGINA = 0.30


def _achar_quadrilatero(image: np.ndarray) -> np.ndarray | None:
    """Maior quadrilatero convexo que ocupe boa parte da foto."""
    altura, largura = image.shape[:2]
    area_foto = float(altura * largura)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bordas = cv2.Canny(gray, 40, 120)
    bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=2)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    melhor: np.ndarray | None = None
    melhor_area = 0.0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_foto * AREA_MINIMA_PAGINA:
            continue
        aprox = cv2.approxPolyDP(contorno, 0.02 * cv2.arcLength(contorno, True), True)
        if len(aprox) != 4 or not cv2.isContourConvex(aprox):
            continue
        if area > melhor_area:
            melhor_area = area
            melhor = aprox.reshape(4, 2).astype("float32")
    return melhor


def _ordenar_pontos(pontos: np.ndarray) -> np.ndarray:
    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1).reshape(-1)
    return np.array(
        [
            pontos[np.argmin(soma)],
            pontos[np.argmin(diferenca)],
            pontos[np.argmax(soma)],
            pontos[np.argmax(diferenca)],
        ],
        dtype="float32",
    )


def _remover_sombra(pagina: np.ndarray) -> np.ndarray:
    """Divide cada canal pelo fundo estimado, deixando o papel branco uniforme."""
    gray = cv2.cvtColor(pagina, cv2.COLOR_BGR2GRAY)
    k = max(31, (min(pagina.shape[:2]) // 15) | 1)
    fundo = cv2.medianBlur(gray, k)
    canais = [
        cv2.divide(canall, fundo, scale=255)
        for canall in cv2.split(pagina)
    ]
    return cv2.merge(canais)


def melhorar_pagina(image: np.ndarray | None, ativo: bool = True) -> tuple[np.ndarray | None, dict]:
    """Aplica recorte + perspectiva + clareamento quando possivel.

    Retorna ``(imagem, info)``. ``info["aplicado"]`` indica se houve
    transformacao; sem deteccao confiavel a foto original volta intacta.
    """
    if not ativo or image is None or image.size == 0:
        return image, {"aplicado": False, "motivo": "desativado"}
    try:
        quad = _achar_quadrilatero(image)
        if quad is None:
            return image, {"aplicado": False, "motivo": "pagina nao detectada"}
        quad = _ordenar_pontos(quad)
        lados = [
            float(np.linalg.norm(quad[i] - quad[(i + 1) % 4]))
            for i in range(4)
        ]
        largura_out = int(max(lados[0], lados[2]))
        altura_out = int(max(lados[1], lados[3]))
        if largura_out < 50 or altura_out < 50:
            return image, {"aplicado": False, "motivo": "pagina pequena demais"}
        destino = np.array(
            [
                [0, 0],
                [largura_out - 1, 0],
                [largura_out - 1, altura_out - 1],
                [0, altura_out - 1],
            ],
            dtype="float32",
        )
        matriz = cv2.getPerspectiveTransform(quad, destino)
        pagina = cv2.warpPerspective(image, matriz, (largura_out, altura_out))
        pagina = _remover_sombra(pagina)
        return pagina, {
            "aplicado": True,
            "recorte_px": [int(quad[0][0]), int(quad[0][1]),
                           largura_out, altura_out],
        }
    except Exception as exc:
        logger.exception("Falha na melhoria de pagina")
        return image, {"aplicado": False, "motivo": f"erro: {exc}"}
