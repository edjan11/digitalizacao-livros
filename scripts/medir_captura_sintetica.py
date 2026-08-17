"""Medicoes sinteticas da captura (M1-T02 complemento sem camera real).

Mede em frames sinteticos: latencia do AutoCaptureController.analisar (detector),
tempo de gravacao do JPG (Q95), e o ritmo de capturas/min numa sequencia de paginas.
Os numeros reais da camera (FPS de preview, jitter da UI) serao coletados na
proxima sessao do operador via eventos capture.sample.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.auto_capture import AutoCaptureController


def _pagina(largura: int, altura: int, seed: int = 1) -> np.ndarray:
    image = np.full((altura, largura, 3), 225, np.uint8)
    m = 0.09
    cv2.rectangle(
        image,
        (int(largura * m), int(altura * m)),
        (int(largura * (1 - m)), int(altura * (1 - m))),
        (250, 250, 250), -1,
    )
    for y in range(int(altura * 0.12), int(altura * 0.88), int(altura * 0.045)):
        cv2.line(image, (int(largura * 0.13), y), (int(largura * 0.87), y), (65, 65, 65), 2)
    cv2.putText(image, f"Numero {6800 + seed}", (int(largura * 0.14), int(altura * 0.13)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2)
    return image


def medir_detector(controller, pagina, vezes=30) -> dict:
    controller.reset()
    controller.analisar(pagina, agora=0.0)  # primeiro frame inclui alocacoes
    tempos = []
    agora = 1.0
    for _ in range(vezes):
        inicio = time.perf_counter()
        controller.analisar(pagina, agora=agora)
        tempos.append((time.perf_counter() - inicio) * 1000)
        agora += 0.05
    return {
        "media_ms": round(float(np.mean(tempos)), 2),
        "max_ms": round(float(np.max(tempos)), 2),
        "p95_ms": round(float(np.percentile(tempos, 95)), 2),
    }


def medir_save(frame, vezes=10) -> dict:
    tempos = []
    for _ in range(vezes):
        inicio = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        tempos.append((time.perf_counter() - inicio) * 1000)
    return {
        "media_ms": round(float(np.mean(tempos)), 2),
        "max_ms": round(float(np.max(tempos)), 2),
        "bytes_media": int(len(buf.tobytes())),
    }


def medir_ritmo(paginas: list[np.ndarray], controller) -> dict:
    """Tempo simulado ate a proxima captura (tempo_estavel 1.2s)."""
    capturas = 0
    agora = 0.0
    total = 0.0
    for indice, pagina in enumerate(paginas):
        if indice:
            controller.analisar(np.full_like(pagina, 35), agora=agora)  # virada
            agora += 0.1
        t0 = agora
        while True:
            analise = controller.analisar(pagina, agora=agora)
            agora += 0.1
            if analise.capturar:
                break
            if agora - t0 > 15:
                raise RuntimeError("nao capturou em 15s")
        total += agora - t0
        capturas += 1
    return {
        "capturas": capturas,
        "media_s_ate_captura": round(total / capturas, 2),
    }


def main() -> None:
    relatorio = {}
    for nome, largura, altura in (("720x960", 960, 720), ("1920x1080", 1920, 1080)):
        pagina = _pagina(largura, altura)
        controller = AutoCaptureController()
        relatorio[nome] = {
            "detector": medir_detector(controller, pagina),
            "save_jpg_q95": medir_save(pagina),
        }
    paginas = [_pagina(960, 720, s) for s in (1, 2, 3)]
    relatorio["ritmo_720x960"] = medir_ritmo(paginas, AutoCaptureController())
    saida = ROOT / ".tmp_captura_sintetica.json"
    saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(relatorio, ensure_ascii=False, indent=1))
    print(f"RESULTADO={saida}")


if __name__ == "__main__":
    main()
