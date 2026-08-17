"""Benchmark heuristica de mao (pontuacao_mao) vs MediaPipe Hand Landmarker.

Cenarios sinteticos realistas (tons de pele clara/escura, punho, mao parcial,
dedos na borda, sem mao). Mede acerto, custo ms/frame e carga. O MediaPipe foi
instalado SOMENTE para este experimento (M3-T02); a regra de producao continua
sendo a heuristica ate a decisao registrada no relatorio.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.auto_capture import pontuacao_mao

LIMIAR_HEURISTICA = 0.20
MODELO_TASK = ROOT / ".tmp_mao" / "hand_landmarker.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _fundo() -> np.ndarray:
    """Pagina sobre mesa: fundo escuro + folha clara com linhas."""
    imagem = np.full((720, 960, 3), 120, np.uint8)
    cv2.rectangle(imagem, (90, 45), (870, 675), (225, 225, 225), -1)
    for y in range(100, 640, 30):
        cv2.line(imagem, (125, y), (835, y), (65, 65, 65), 2)
    return imagem


def _mao(pele: tuple[int, int, int], area: tuple[int, int, int, int],
         dedos: bool = False) -> np.ndarray:
    """Bloco de pele (ou formato de mao com dedos) sobre o fundo."""
    imagem = _fundo()
    x0, y0, x1, y1 = area
    if dedos:
        cv2.rectangle(imagem, (x0, y0 + 60), (x1, y1), pele, -1)  # palma
        largura = (x1 - x0) // 5
        for i in range(5):
            dx = x0 + i * largura
            cv2.rectangle(imagem, (dx, y0), (dx + largura // 2, y0 + 90), pele, -1)
    else:
        cv2.rectangle(imagem, (x0, y0), (x1, y1), pele, -1)
    return imagem


def _sem_mao() -> np.ndarray:
    return _fundo()


def cenarios() -> list[tuple[str, np.ndarray, bool]]:
    pele_clara = (185, 215, 245)     # BGR: tom claro
    pele_media = (140, 170, 210)
    pele_escura = (85, 115, 160)     # tom escuro
    return [
        ("sem_mao", _sem_mao(), False),
        ("mao_clara_30pct", _mao(pele_clara, (40, 40, 500, 680)), True),
        ("mao_media_30pct", _mao(pele_media, (40, 40, 500, 680)), True),
        ("mao_escura_30pct", _mao(pele_escura, (40, 40, 500, 680)), True),
        ("punho_20pct", _mao(pele_media, (250, 180, 620, 560)), True),
        ("mao_parcial_borda", _mao(pele_media, (0, 40, 430, 680)), True),
        ("dedos_na_borda", _mao(pele_media, (10, 300, 420, 620), dedos=True), True),
        ("mao_clara_50pct", _mao(pele_clara, (40, 40, 700, 680)), True),
        ("mao_escura_50pct", _mao(pele_escura, (40, 40, 700, 680)), True),
    ]


def carregar_mediapipe():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python

    MODELO_TASK.parent.mkdir(parents=True, exist_ok=True)
    if not MODELO_TASK.exists():
        import urllib.request
        urllib.request.urlretrieve(URL_MODELO, str(MODELO_TASK))
    options = mp_python.vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODELO_TASK)),
        running_mode=mp_python.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return mp_python.vision.HandLandmarker.create_from_options(options)


def main() -> None:
    import mediapipe as mp
    mp_detector = carregar_mediapipe()
    from mediapipe.tasks import python as mp_python

    print("Cenarios | esperado | heuristica | mediapipe | heur_ms | mp_ms")
    resultados = []
    for nome, frame, esperado in cenarios():
        t0 = time.perf_counter()
        score = pontuacao_mao(frame)
        heur_ms = (time.perf_counter() - t0) * 1000
        heur_presente = score >= LIMIAR_HEURISTICA

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )
        t0 = time.perf_counter()
        resultado = mp_detector.detect(imagem_mp)
        mp_ms = (time.perf_counter() - t0) * 1000
        mp_presente = bool(resultado.hand_landmarks)

        acerto_heur = heur_presente == esperado
        acerto_mp = mp_presente == esperado
        resultados.append({
            "cenario": nome, "esperado": esperado,
            "heuristica": heur_presente, "heuristica_score": round(score, 3),
            "mediapipe": mp_presente,
            "heur_ms": round(heur_ms, 1), "mp_ms": round(mp_ms, 1),
            "acerto_heur": acerto_heur, "acerto_mp": acerto_mp,
        })
        print(
            f"{nome:<18} | {str(esperado):<8} | {str(heur_presente):<10} | "
            f"{str(mp_presente):<9} | {heur_ms:6.1f} | {mp_ms:6.1f}",
            flush=True,
        )

    n = len(resultados)
    acertos_heur = sum(1 for r in resultados if r["acerto_heur"])
    acertos_mp = sum(1 for r in resultados if r["acerto_mp"])
    heur_ms_med = sum(r["heur_ms"] for r in resultados) / n
    mp_ms_med = sum(r["mp_ms"] for r in resultados) / n
    resumo = {
        "cenarios": n,
        "acertos_heuristica": acertos_heur,
        "acertos_mediapipe": acertos_mp,
        "heuristica_ms_medio": round(heur_ms_med, 1),
        "mediapipe_ms_medio": round(mp_ms_med, 1),
        "limiar_heuristica": LIMIAR_HEURISTICA,
        "detalhes": resultados,
    }
    saida = ROOT / ".tmp_benchmark_mao.json"
    saida.write_text(
        __import__("json").dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nAcertos: heuristica {acertos_heur}/{n} | mediapipe {acertos_mp}/{n}")
    print(f"Tempo medio/frame: heuristica {heur_ms_med:.1f} ms | mediapipe {mp_ms_med:.1f} ms")
    print(f"RESULTADO={saida}")


if __name__ == "__main__":
    main()
