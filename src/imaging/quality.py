from __future__ import annotations

import logging

import cv2
import numpy as np
from pathlib import Path

from ..capture.auto_capture import pontuacao_mao
from .document import pagina_cortada_na_borda

logger = logging.getLogger(__name__)

LAPLACIAN_BLUR_THRESHOLD = 100
DARK_PERCENTILE = 40
OVEREXPOSED_PCT = 30
SKEW_MAX = 3.0
EMPTY_STD_THRESHOLD = 15
BORDER_FRACTION = 0.05


def detectar_foco(image: np.ndarray) -> tuple[float, str]:
    if image is None or image.size == 0:
        return 0.0, "erro"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        return lap_var, "erro_grave"
    if lap_var < LAPLACIAN_BLUR_THRESHOLD:
        return lap_var, "revisar"
    if lap_var < 200:
        return lap_var, "aviso"
    return lap_var, "ok"


def detectar_exposicao(image: np.ndarray) -> tuple[float, str]:
    if image is None or image.size == 0:
        return 0.0, "erro"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    median = np.percentile(gray, 50)
    over = (gray > 250).mean() * 100
    if median < DARK_PERCENTILE:
        return median, "revisar"
    if over > OVEREXPOSED_PCT:
        return median, "revisar"
    if median < 80 or over > 20:
        return median, "aviso"
    return median, "ok"


def detectar_enquadramento(image: np.ndarray) -> str:
    if image is None or image.size == 0:
        return "erro"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    border_top = gray[: max(1, int(h * BORDER_FRACTION)), :].mean()
    border_bottom = gray[min(h - 1, int(h * (1 - BORDER_FRACTION))):, :].mean()
    border_left = gray[:, : max(1, int(w * BORDER_FRACTION))].mean()
    border_right = gray[:, min(w - 1, int(w * (1 - BORDER_FRACTION))):].mean()
    borders = [border_top, border_bottom, border_left, border_right]
    dark_borders = sum(1 for b in borders if b < 30)
    if dark_borders >= 2:
        return "revisar"
    if dark_borders >= 1:
        return "aviso"
    return "ok"


def detectar_vazia(image: np.ndarray) -> bool:
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(gray.std()) < EMPTY_STD_THRESHOLD


def detectar_dobra_grande(image: np.ndarray) -> tuple[float, bool]:
    """Procura uma dobra diagonal forte sem confundir a curvatura do livro.

    Uma linha so conta quando suas duas pontas ficam dentro da pagina e ha
    diferenca de iluminacao dos dois lados. Isso descarta bordas do livro,
    pautas impressas e a maior parte da escrita cursiva.
    """
    if image is None or image.size == 0:
        return 0.0, False
    h, w = image.shape[:2]
    escala = min(1.0, 360 / max(1, w))
    pequena = cv2.resize(
        image,
        (max(1, int(w * escala)), max(1, int(h * escala))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY) if pequena.ndim == 3 else pequena
    suave = cv2.GaussianBlur(gray, (7, 7), 0)
    bordas = cv2.Canny(suave, 55, 150)
    minimo = min(gray.shape[:2])
    linhas = cv2.HoughLinesP(
        bordas,
        1,
        np.pi / 180,
        threshold=55,
        minLineLength=max(40, int(minimo * 0.28)),
        maxLineGap=10,
    )
    maior = 0.0
    if linhas is not None:
        altura, largura = gray.shape
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            angulo = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))) % 180
            distancia_eixo = min(angulo, abs(90 - angulo), abs(180 - angulo))
            if distancia_eixo < 14:
                continue
            # As duas pontas devem estar dentro da pagina; uma linha da borda
            # curva do livro nao e uma dobra.
            pontos_rel = (
                x1 / largura, y1 / altura, x2 / largura, y2 / altura
            )
            if not (
                0.10 <= pontos_rel[0] <= 0.90
                and 0.08 <= pontos_rel[1] <= 0.92
                and 0.10 <= pontos_rel[2] <= 0.90
                and 0.08 <= pontos_rel[3] <= 0.92
            ):
                continue
            comprimento = float(np.hypot(x2 - x1, y2 - y1)) / max(1, minimo)
            if comprimento < 0.30:
                continue

            dx, dy = float(x2 - x1), float(y2 - y1)
            norma = max(1.0, float(np.hypot(dx, dy)))
            nx, ny = -dy / norma, dx / norma
            diferencas = []
            for t in np.linspace(0.15, 0.85, 18):
                px = x1 + dx * t
                py = y1 + dy * t
                deslocamento = 5
                ax = int(round(px + nx * deslocamento))
                ay = int(round(py + ny * deslocamento))
                bx = int(round(px - nx * deslocamento))
                by = int(round(py - ny * deslocamento))
                if 0 <= ax < largura and 0 <= bx < largura and 0 <= ay < altura and 0 <= by < altura:
                    diferencas.append(abs(int(suave[ay, ax]) - int(suave[by, bx])))
            contraste = float(np.median(diferencas)) if diferencas else 0.0
            score = comprimento * min(1.0, contraste / 24.0)
            maior = max(maior, score)
    return round(maior, 3), maior >= 0.34


def avaliar_qualidade(image: np.ndarray, exigir_margens: bool = False) -> dict:
    # Qualidade nao precisa percorrer todos os pixels de uma fotografia de
    # dezenas de megapixels. A copia reduzida deixa a captura responsiva e o
    # original nunca e alterado.
    if image is not None and image.size and image.shape[1] > 1200:
        escala = 1200 / image.shape[1]
        analise = cv2.resize(
            image,
            (1200, max(1, int(image.shape[0] * escala))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analise = image
    foco_val, foco_status = detectar_foco(analise)
    expo_val, expo_status = detectar_exposicao(analise)
    enq_status = detectar_enquadramento(analise)
    vazia = detectar_vazia(analise)
    oclusao = pontuacao_mao(analise)
    dobra_score, dobra = detectar_dobra_grande(analise)
    corte = pagina_cortada_na_borda(analise) if exigir_margens else False
    statuses = [foco_status, expo_status, enq_status]
    if vazia:
        status_geral = "erro_grave"
    elif "erro_grave" in statuses:
        status_geral = "erro_grave"
    elif "revisar" in statuses:
        status_geral = "revisar"
    elif "aviso" in statuses:
        status_geral = "aviso"
    else:
        status_geral = "ok"
    motivos_refazer = []
    if foco_status in ("erro_grave", "revisar"):
        motivos_refazer.append("imagem desfocada")
    if expo_status == "revisar":
        motivos_refazer.append("exposicao inadequada")
    if enq_status == "revisar":
        motivos_refazer.append("enquadramento insuficiente")
    if oclusao >= 0.20:
        motivos_refazer.append("mao ou objeto cobrindo a pagina")
    if dobra:
        motivos_refazer.append("dobra grande sobre o documento")
    if corte:
        motivos_refazer.append("pagina ou margem de averbacoes cortada")
    if motivos_refazer and status_geral == "ok":
        status_geral = "revisar"
    return {
        "foco_valor": round(foco_val, 2),
        "foco_status": foco_status,
        "exposicao_valor": round(expo_val, 2),
        "exposicao_status": expo_status,
        "enquadramento_status": enq_status,
        "vazia": vazia,
        "oclusao_valor": round(oclusao, 3),
        "dobra_valor": dobra_score,
        "corte_detectado": corte,
        "motivos_refazer": motivos_refazer,
        "repetir_captura": bool(motivos_refazer),
        "status_geral": status_geral,
    }
