"""Benchmark do Qwen2.5-VL-3B via Ollama nos mesmos 20 recortes do A-07.

Comparacao justa com o benchmark do Qwen2-VL-2B local: mesma pasta de
recortes (.tmp_qwen_20_line), mesmo gabarito e mesmas metricas. O parametro
--max-width reduz a copia em memoria antes de enviar a imagem (o arquivo
armazenado nunca e alterado), medindo o impacto do item "menos pixels para a IA".
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalizar(texto: str | None) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", texto.lower())


def distancia(a: str, b: str) -> int:
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(
                atual[-1] + 1,
                anterior[j] + 1,
                anterior[j - 1] + (ca != cb),
            ))
        anterior = atual
    return anterior[-1]


def similarity(pred: str, gold: str) -> float:
    a, b = normalizar(pred), normalizar(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - distancia(a, b) / max(len(a), len(b)))


INSTRUCAO = (
    "A imagem e um recorte de um registro civil brasileiro. "
    "Leia SOMENTE o nome manuscrito que vem depois das palavras "
    "impressas 'que recebeu o nome de'. Ignore todo o resto. "
    "Responda apenas com o nome completo, sem explicacao. "
    "Se uma letra estiver ilegive, preserve a duvida e nao invente."
)


def gerar(modelo: str, imagem: np.ndarray, num_thread: int) -> tuple[str, float]:
    dados = cv2.imencode(".jpg", imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]
    payload = {
        "model": modelo,
        "prompt": INSTRUCAO,
        "images": [base64.b64encode(dados).decode("ascii")],
        "stream": False,
        "options": {"num_thread": num_thread},
    }
    requisicao = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    inicio = time.perf_counter()
    with urllib.request.urlopen(requisicao, timeout=600) as resp:
        corpo = json.loads(resp.read().decode("utf-8"))
    texto = str(corpo.get("response") or "").strip()
    # O Ollama reporta o tempo de avaliacao do modelo; o cronometro mede o
    # ciclo completo (rede + prefill + decode).
    tempo_modelo = float(corpo.get("total_duration") or 0) / 1e9
    return texto, tempo_modelo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".tmp_qwen_20_line")
    ap.add_argument("--gold", default="scripts/gabarito_qwen_a07_20.json")
    ap.add_argument("--output", default=".tmp_ollama_20_result.json")
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--max-width", type=int, default=0,
                    help="0 = envia o recorte original; >0 reduz a largura em memoria")
    args = ap.parse_args()

    gold_data = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    gold_by_term = {str(k): v for k, v in gold_data.items()}

    files = sorted(
        (p for p in Path(args.input).glob("*.jpg") if p.name != "montagem.jpg"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    results = []
    for path in files:
        term = path.stem.split("_")[-1]
        dados = np.fromfile(str(path), dtype=np.uint8)
        imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
        original_w = int(imagem.shape[1])
        if args.max_width and imagem.shape[1] > args.max_width:
            escala = args.max_width / imagem.shape[1]
            imagem = cv2.resize(
                imagem,
                (args.max_width, max(1, int(imagem.shape[1] * escala))),
                interpolation=cv2.INTER_AREA,
            )
        inicio = time.perf_counter()
        error = None
        prediction = ""
        tempo_modelo = 0.0
        try:
            prediction, tempo_modelo = gerar(args.model, imagem, args.threads)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - inicio) * 1000
        gold = gold_by_term.get(term)
        results.append({
            "termo": int(term),
            "arquivo": path.name,
            "largura_original": original_w,
            "largura_enviada": int(imagem.shape[1]),
            "predicao": prediction,
            "gold": gold,
            "similaridade": similarity(prediction, gold) if gold else None,
            "tempo_ms": round(elapsed_ms, 1),
            "tempo_modelo_s": round(tempo_modelo, 2),
            "erro": error,
        })
        print(
            f"{term}: {prediction or '—'} | {round(elapsed_ms / 1000, 2)} s"
            + (f" | ERRO {error}" if error else ""),
            flush=True,
        )

    scored = [r for r in results if r["gold"]]
    exact = sum(normalizar(r["predicao"]) == normalizar(r["gold"]) for r in scored)
    summary = {
        "motor": "ollama-" + args.model,
        "max_width": args.max_width or "original",
        "threads": args.threads,
        "registros": len(results),
        "avaliados": len(scored),
        "exatos": exact,
        "taxa_exata": exact / len(scored) if scored else 0.0,
        "similaridade_media": sum(r["similaridade"] for r in scored) / len(scored)
        if scored
        else 0.0,
        "distancia_media": round(
            sum(distancia(normalizar(r["predicao"]), normalizar(r["gold"])) for r in scored)
            / len(scored), 2,
        ) if scored else None,
        "tempo_medio_s": round(sum(r["tempo_ms"] for r in results) / len(results) / 1000, 2)
        if results
        else 0.0,
        "resultados": results,
    }
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: summary[k] for k in summary if k != "resultados"}, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
