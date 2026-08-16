from __future__ import annotations

"""Regiões normalizadas dos assentos dentro de uma fotografia."""

from dataclasses import dataclass

import numpy as np


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class RecordRegion:
    indice: int
    total: int
    bbox: BBox
    nome_bbox: BBox


def _indice_valido(indice: int, total: int) -> tuple[int, int]:
    total = max(1, int(total))
    return max(0, min(int(indice), total - 1)), total


def bbox_registro(indice: int, total: int) -> BBox:
    """Corpo do assento, sem margem esquerda nem coluna de averbações.

    A faixa e generosa de proposito: inclui um pouco da margem esquerda e
    alcanca quase a divisoria de averbacoes, reduzindo cortes que atingem
    ascendentes/descendentes quando a camera deixa mais ou menos margem.
    """
    indice, total = _indice_valido(indice, total)
    if total == 2:
        y1, y2 = ((0.004, 0.496), (0.490, 0.996))[indice]
    else:
        bloco = 1.0 / total
        y1 = indice * bloco + 0.004
        y2 = (indice + 1) * bloco - 0.004
    return (0.045, y1, 0.76, y2)


def bbox_faixa_nome(indice: int, total: int) -> BBox:
    """Faixa conservadora do campo ``que recebeu o nome de`` no A-07."""
    _x1, y1, x2, y2 = bbox_registro(indice, total)
    altura = y2 - y1
    return (0.14, y1 + altura * 0.25, x2, y1 + altura * 0.44)


def bbox_data_registro(indice: int, total: int) -> BBox:
    """Cabeçalho manuscrito com a data em que o ato foi lavrado."""
    _x1, y1, x2, y2 = bbox_registro(indice, total)
    altura = y2 - y1
    # A linha começa depois do rótulo impresso "Em" e termina antes da
    # divisória de averbações. Incluímos a segunda linha, usada quando a data
    # continua em "e oitenta e três", sem alcançar o assento seguinte.
    return (0.14, y1 + altura * 0.02, x2, y1 + altura * 0.20)


def bbox_numero_termo(indice: int, total: int) -> BBox:
    """Recorte pequeno do numero manuscrito no cabecalho do assento."""
    _x1, y1, _x2, y2 = bbox_registro(indice, total)
    altura = y2 - y1
    return (0.035, y1 + altura * 0.035, 0.19, y1 + altura * 0.18)


def bbox_linha_nome(indice: int, total: int) -> BBox:
    """Linha estreita que contém o rótulo e o nome manuscrito.

    A faixa ampla era útil para diagnóstico, mas incluía as linhas de horário,
    sexo e filiação. O A-07 mantém o campo ``que recebeu o nome de`` na metade
    inferior dessa faixa; reduzir aqui evita que o Qwen/Tesseract misture
    campos vizinhos.
    """
    _x1, y1, x2, y2 = bbox_faixa_nome(indice, total)
    altura = y2 - y1
    # O texto impresso "que recebeu o nome de" ocupa a parte inicial da
    # linha. Começamos depois dele, preservando a escrita manuscrita e sem
    # incluir a coluna de averbações.
    # A linha começa antes da posição média da faixa: iniciar em 0.54 corta
    # os ascendentes de nomes como ``Eduardo``.  Este intervalo inclui a linha
    # inteira e ainda termina antes da linha de filiação seguinte.
    return (0.28, y1 + altura * 0.26, x2, y1 + altura * 0.88)


def regiao_registro(indice: int, total: int) -> RecordRegion:
    indice, total = _indice_valido(indice, total)
    return RecordRegion(
        indice=indice,
        total=total,
        bbox=bbox_registro(indice, total),
        nome_bbox=bbox_faixa_nome(indice, total),
    )


def recortar_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    altura, largura = image.shape[:2]
    x1 = max(0, min(largura - 1, int(float(bbox[0]) * largura)))
    y1 = max(0, min(altura - 1, int(float(bbox[1]) * altura)))
    x2 = max(x1 + 1, min(largura, int(float(bbox[2]) * largura)))
    y2 = max(y1 + 1, min(altura, int(float(bbox[3]) * altura)))
    return image[y1:y2, x1:x2]


def recortar_registro(image: np.ndarray, indice: int, total: int) -> np.ndarray:
    return recortar_bbox(image, bbox_registro(indice, total))


def recortar_faixa_nome(image: np.ndarray, indice: int, total: int) -> np.ndarray:
    return recortar_bbox(image, bbox_faixa_nome(indice, total))


def bbox_corresponde_registro(
    bbox: BBox | list[float] | tuple[float, ...],
    indice: int,
    total: int,
    *,
    tolerancia: float = 0.025,
) -> bool:
    """Rejeita associação a outro assento ou inclusão de averbações."""
    if len(bbox) != 4:
        return False
    esperado = bbox_registro(indice, total)
    atual = tuple(float(valor) for valor in bbox)
    return (
        atual[0] >= esperado[0] - tolerancia
        and atual[2] <= esperado[2] + tolerancia
        and abs(atual[1] - esperado[1]) <= tolerancia
        and abs(atual[3] - esperado[3]) <= tolerancia
        and atual[0] < atual[2]
        and atual[1] < atual[3]
    )


def bbox_contido_no_registro(
    bbox: BBox | list[float] | tuple[float, ...],
    indice: int,
    total: int,
    *,
    tolerancia: float = 0.015,
) -> bool:
    """Valida uma seleção manual menor dentro do assento correto."""
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(valor) for valor in bbox)
    ex1, ey1, ex2, ey2 = bbox_registro(indice, total)
    return (
        x1 >= ex1 - tolerancia
        and x2 <= ex2 + tolerancia
        and y1 >= ey1 - tolerancia
        and y2 <= ey2 + tolerancia
        and x1 < x2
        and y1 < y2
    )
