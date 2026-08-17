"""Analise exploratoria para calibrar detectar_glare (M3-T01).

Mede nas fotos reais do A-07 a fracao de 'blob claro e lavado' compacto:
area do maior componente conexo de pixels >=250 com desvio local baixo,
sobre a area total da imagem. Glare real = blob compacto grande.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def blob_lavado(image: np.ndarray, minimo_fracao: float = 0.01) -> float:
    h, w = image.shape[:2]
    escala = min(1.0, 800 / max(1, w))
    pequena = cv2.resize(
        image, (max(1, int(w * escala)), max(1, int(h * escala))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY) if pequena.ndim == 3 else pequena
    alto = (gray >= 250).astype(np.uint8)
    g = gray.astype(np.float32)
    media = cv2.boxFilter(g, ddepth=-1, ksize=(15, 15))
    media2 = cv2.boxFilter(g * g, ddepth=-1, ksize=(15, 15))
    desvio = np.sqrt(np.maximum(media2 - media * media, 0))
    lavado = ((alto == 1) & (desvio < 12)).astype(np.uint8)
    n, _lbl, stats, _cent = cv2.connectedComponentsWithStats(lavado, connectivity=8)
    if n <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    maior = int(areas.max())
    return maior / float(max(1, gray.shape[0] * gray.shape[1]))


def main() -> None:
    fotos = sorted(Path(r"D:\A - 07").rglob("*.jpg"))
    valores = []
    for caminho in fotos:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
        imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
        if imagem is None:
            continue
        valores.append((blob_lavado(imagem), caminho.name))
    valores.sort(reverse=True)
    print(f"{len(valores)} fotos")
    print("Top 10 maiores blobs lavados:")
    for fracao, nome in valores[:10]:
        print(f"  {fracao:.4f}  {nome}")
    fracs = [v[0] for v in valores]
    for limite in (0.01, 0.02, 0.03, 0.05, 0.08):
        n = sum(1 for f in fracs if f >= limite)
        print(f"acima de {limite:.2f}: {n} fotos ({n/len(fracs)*100:.1f}%)")


if __name__ == "__main__":
    main()
