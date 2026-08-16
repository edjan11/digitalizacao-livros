from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RectificationResult:
    image: np.ndarray
    applied: bool
    confidence: float
    angle_degrees: float
    left_line: tuple[float, float] | None = None
    right_line: tuple[float, float] | None = None
    reason: str = ""


def detectar_quadrilatero_pagina(image: np.ndarray, largura: int = 600) -> np.ndarray | None:
    """Retorna os quatro pontos normalizados de uma pagina bem delimitada."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    escala = min(1.0, largura / max(1, w))
    pequena = cv2.resize(
        image,
        (max(1, int(w * escala)), max(1, int(h * escala))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY) if pequena.ndim == 3 else pequena
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    area_frame = float(gray.shape[0] * gray.shape[1])
    melhor = None
    melhor_area = 0.0
    for baixo, alto in ((50, 150), (70, 200), (30, 100)):
        bordas = cv2.Canny(gray, baixo, alto)
        bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=1)
        contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(contorno)
            if area < area_frame * 0.30:
                break
            perimetro = cv2.arcLength(contorno, True)
            poligono = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            if len(poligono) != 4 or not cv2.isContourConvex(poligono):
                continue
            if area > melhor_area:
                pontos = poligono.reshape(4, 2).astype(np.float32)
                pontos[:, 0] /= gray.shape[1]
                pontos[:, 1] /= gray.shape[0]
                melhor = pontos
                melhor_area = area
        if melhor is not None:
            break
    return melhor


def pagina_cortada_na_borda(image: np.ndarray, margem: float = 0.015) -> bool:
    pontos = detectar_quadrilatero_pagina(image)
    if pontos is None:
        return False
    min_x, max_x = float(pontos[:, 0].min()), float(pontos[:, 0].max())
    min_y, max_y = float(pontos[:, 1].min()), float(pontos[:, 1].max())
    return min_x <= margem or min_y <= margem or max_x >= 1.0 - margem or max_y >= 1.0 - margem


def _linhas_formulario(image: np.ndarray, largura: int = 800):
    altura_original, largura_original = image.shape[:2]
    escala = min(1.0, largura / max(1, largura_original))
    pequena = cv2.resize(
        image,
        (max(1, int(largura_original * escala)), max(1, int(altura_original * escala))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY) if pequena.ndim == 3 else pequena
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    minimo = max(80, int(min(gray.shape[:2]) * 0.16))
    linhas = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=75,
        minLineLength=minimo,
        maxLineGap=40,
    )
    return pequena, [] if linhas is None else linhas.reshape(-1, 4)


def _angulo_horizontal(linhas, largura: int) -> tuple[float, float]:
    candidatos: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in linhas:
        dx, dy = float(x2 - x1), float(y2 - y1)
        comprimento = math.hypot(dx, dy)
        if comprimento < largura * 0.30 or abs(dx) < 1:
            continue
        angulo = math.degrees(math.atan2(dy, dx))
        if abs(angulo) <= 5.0:
            candidatos.append((angulo, comprimento))
    if len(candidatos) < 5:
        return 0.0, 0.0
    ordenados = sorted(candidatos, key=lambda item: item[0])
    total = sum(item[1] for item in ordenados)
    acumulado = 0.0
    mediana = 0.0
    for angulo, peso in ordenados:
        acumulado += peso
        if acumulado >= total / 2:
            mediana = angulo
            break
    confianca = min(1.0, total / max(1.0, largura * 25.0))
    return mediana, confianca


def _linha_vertical_por_faixa(
    linhas,
    largura: int,
    altura: int,
    minimo_rel: float,
    maximo_rel: float,
) -> tuple[float, float, float] | None:
    pontos: list[tuple[float, float, float]] = []
    comprimento_total = 0.0
    for x1, y1, x2, y2 in linhas:
        dx, dy = float(x2 - x1), float(y2 - y1)
        comprimento = math.hypot(dx, dy)
        if comprimento < altura * 0.18 or abs(dy) < 1:
            continue
        angulo = abs(math.degrees(math.atan2(dy, dx)))
        if abs(angulo - 90.0) > 9.0:
            continue
        meio_x = (float(x1) + float(x2)) / 2
        if not minimo_rel * largura <= meio_x <= maximo_rel * largura:
            continue
        pontos.extend(((float(y1), float(x1), comprimento), (float(y2), float(x2), comprimento)))
        comprimento_total += comprimento
    if len(pontos) < 4 or comprimento_total < altura * 0.35:
        return None
    ys = np.array([p[0] for p in pontos], dtype=np.float64)
    xs = np.array([p[1] for p in pontos], dtype=np.float64)
    pesos = np.sqrt(np.array([p[2] for p in pontos], dtype=np.float64))
    a, b = np.polyfit(ys, xs, 1, w=pesos)
    return float(a), float(b), min(1.0, comprimento_total / max(1.0, altura * 1.5))


def retificar_formulario(image: np.ndarray) -> RectificationResult:
    """Endireita uma cópia usando linhas impressas, sem alterar o original.

    A rotação nivela as pautas horizontais. Depois, duas linhas verticais do
    formulário são levadas a posições constantes, removendo a perspectiva.
    Se não houver evidência suficiente, devolve uma cópia sem transformação.
    """
    if image is None or image.size == 0:
        return RectificationResult(image=image, applied=False, confidence=0.0, angle_degrees=0.0, reason="imagem vazia")
    original = image.copy()
    pequena, linhas = _linhas_formulario(original)
    angulo, confianca_horizontal = _angulo_horizontal(linhas, pequena.shape[1])
    if confianca_horizontal < 0.35 or abs(angulo) > 4.0:
        return RectificationResult(
            image=original,
            applied=False,
            confidence=confianca_horizontal,
            angle_degrees=angulo,
            reason="linhas horizontais insuficientes",
        )

    altura, largura = original.shape[:2]
    matriz = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, 1.0)
    girada = cv2.warpAffine(
        original,
        matriz,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    pequena_girada, linhas_giradas = _linhas_formulario(girada)
    gh, gw = pequena_girada.shape[:2]
    esquerda = _linha_vertical_por_faixa(linhas_giradas, gw, gh, 0.12, 0.30)
    direita = _linha_vertical_por_faixa(linhas_giradas, gw, gh, 0.62, 0.82)
    if esquerda is None or direita is None:
        return RectificationResult(
            image=girada,
            applied=abs(angulo) >= 0.08,
            confidence=min(0.69, confianca_horizontal),
            angle_degrees=angulo,
            reason="perspectiva não confirmada; aplicado apenas alinhamento horizontal",
        )

    escala_x = largura / gw
    escala_y = altura / gh
    ae, be, ce = esquerda
    ad, bd, cd = direita
    esquerda_topo = be * escala_x
    esquerda_base = (ae * gh + be) * escala_x
    direita_topo = bd * escala_x
    direita_base = (ad * gh + bd) * escala_x
    if min(direita_topo - esquerda_topo, direita_base - esquerda_base) < largura * 0.35:
        return RectificationResult(
            image=girada,
            applied=abs(angulo) >= 0.08,
            confidence=min(0.60, confianca_horizontal),
            angle_degrees=angulo,
            reason="linhas verticais incompatíveis; aplicado apenas alinhamento horizontal",
        )
    destino_esquerda = (esquerda_topo + esquerda_base) / 2
    destino_direita = (direita_topo + direita_base) / 2
    origem = np.float32([
        [esquerda_topo, 0],
        [direita_topo, 0],
        [direita_base, altura - 1],
        [esquerda_base, altura - 1],
    ])
    destino = np.float32([
        [destino_esquerda, 0],
        [destino_direita, 0],
        [destino_direita, altura - 1],
        [destino_esquerda, altura - 1],
    ])
    perspectiva = cv2.getPerspectiveTransform(origem, destino)
    retificada = cv2.warpPerspective(
        girada,
        perspectiva,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    confianca = min(1.0, 0.45 * confianca_horizontal + 0.275 * ce + 0.275 * cd)
    return RectificationResult(
        image=retificada,
        applied=True,
        confidence=confianca,
        angle_degrees=angulo,
        left_line=(esquerda_topo / largura, esquerda_base / largura),
        right_line=(direita_topo / largura, direita_base / largura),
        reason="formulário alinhado pelas pautas e bordas verticais",
    )
