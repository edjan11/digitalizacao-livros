"""Roteiro deterministico de usabilidade para um lote de digitalizacao.

Nao abre a camera fisica: usa frames falsos para exercitar as mesmas decisoes
do AutoCaptureController. O objetivo e detectar regressao antes de um lote
real, principalmente duplicata, mao, foco, dobra e pagina cortada.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.auto_capture import AutoCaptureController
from src.imaging.quality import avaliar_qualidade


def pagina(numero: int = 6801) -> np.ndarray:
    imagem = np.full((720, 960, 3), 225, np.uint8)
    cv2.rectangle(imagem, (90, 45), (870, 675), (250, 250, 250), -1)
    cv2.rectangle(imagem, (90, 45), (870, 675), (25, 25, 25), 5)
    for y in range(100, 640, 30):
        cv2.line(imagem, (125, y), (835, y), (65, 65, 65), 2)
    cv2.putText(
        imagem, f"Numero {numero}", (135, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2,
    )
    return imagem


def com_mao(imagem: np.ndarray) -> np.ndarray:
    resultado = imagem.copy()
    cv2.rectangle(resultado, (0, 0), (420, 719), (85, 135, 205), -1)
    return resultado


def cortada_na_borda() -> np.ndarray:
    imagem = np.full((720, 960, 3), 25, np.uint8)
    cv2.rectangle(imagem, (90, 45), (959, 675), (240, 240, 240), -1)
    cv2.rectangle(imagem, (90, 45), (959, 675), (10, 10, 10), 5)
    for y in range(100, 640, 30):
        cv2.line(imagem, (125, y), (950, y), (60, 60, 60), 2)
    return imagem


def dobrada(imagem: np.ndarray) -> np.ndarray:
    resultado = imagem.copy()
    triangulo = np.array([[220, 170], [690, 570], [760, 170]], np.int32)
    cv2.fillConvexPoly(resultado, triangulo, (170, 170, 170))
    cv2.line(resultado, (220, 170), (690, 570), (70, 70, 70), 6)
    return resultado


def main() -> int:
    boa = pagina()
    casos = {
        "boa": boa,
        "mao": com_mao(boa),
        "desfocada": cv2.GaussianBlur(boa, (31, 31), 0),
        "dobrada": dobrada(boa),
        "averbacao_cortada": cortada_na_borda(),
    }
    relatorio = {}
    for nome, imagem in casos.items():
        qualidade = avaliar_qualidade(
            imagem, exigir_margens=(nome == "averbacao_cortada")
        )
        relatorio[nome] = {
            "status": qualidade["status_geral"],
            "refazer": qualidade["repetir_captura"],
            "motivos": qualidade["motivos_refazer"],
        }

    controlador = AutoCaptureController(tempo_estavel=0.4)
    assert not controlador.analisar(boa, agora=0.0).capturar
    assert not controlador.analisar(boa, agora=0.5).capturar
    assert controlador.analisar(boa, agora=0.9).capturar
    assert controlador.analisar(boa, agora=1.0).status == "Troque a pagina"
    assert not controlador.analisar(com_mao(boa), agora=1.1).capturar
    assert controlador.analisar(np.full_like(boa, 35), agora=1.2).capturar is False
    assert not controlador.analisar(pagina(6802), agora=1.3).capturar

    assert relatorio["boa"]["refazer"] is False
    assert relatorio["mao"]["refazer"] is True
    assert relatorio["desfocada"]["refazer"] is True
    assert relatorio["dobrada"]["refazer"] is True
    assert relatorio["averbacao_cortada"]["refazer"] is True
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    print("USABILIDADE: OK — pagina boa, duplicata, mao, foco, dobra e corte cobertos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
