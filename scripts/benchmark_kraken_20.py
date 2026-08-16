"""Benchmark Kraken on the same A-07 name regions used by the Qwen test.

This intentionally uses a fixed, reproducible one-line window.  The downloaded
TraPrInq model is a recognition model (Portuguese handwriting, 16th--19th c.),
so a small deterministic line window avoids mixing the model's score with a
second page-segmentation failure.  It is still a fair recognition comparison
because every engine receives the same source crop and no overlay is sent.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

from PIL import Image

from kraken import containers, rpred
from kraken.lib import models


def normalizar(texto: str) -> str:
    text = unicodedata.normalize("NFKD", texto or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def distancia(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(pred: str, gold: str) -> float:
    a, b = normalizar(pred), normalizar(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - distancia(a, b) / max(len(a), len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".tmp_qwen_20_line")
    ap.add_argument("--model", default="models/kraken/TraPrInq.mlmodel")
    ap.add_argument("--gold", default="scripts/gabarito_qwen_a07_20.json")
    ap.add_argument("--output", default=".tmp_kraken_20_result.json")
    args = ap.parse_args()

    gold_data = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if isinstance(gold_data, dict):
        gold_by_term = {str(k): v for k, v in gold_data.items()}
    else:
        gold_by_term = {str(x["termo"]): x.get("nome") for x in gold_data}
    model = models.load_any(args.model, device="cpu")
    results = []
    files = sorted(Path(args.input).glob("*.jpg"))
    for path in files:
        match = re.search(r"_(\d+)\.jpg$", path.name)
        if not match:
            continue
        term = match.group(1)
        full = Image.open(path).convert("L")
        # Same name ROI as Qwen, with a fixed one-line window around the
        # printed “que recebeu o nome de” field.  Exclude the last two pixels
        # so Kraken's polygon is strictly inside the image bounds.
        line = full.crop((0, 50, full.width, min(full.height, 210)))
        w, h = line.size
        baseline_y = int(h * 0.63)
        bounds = containers.BaselineLine(
            id="0",
            baseline=[(1, baseline_y), (w - 2, baseline_y)],
            boundary=[(1, 1), (w - 2, 1), (w - 2, h - 2), (1, h - 2)],
        )
        seg = containers.Segmentation(
            type="baselines",
            imagename=str(path),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=[bounds],
        )
        started = time.perf_counter()
        error = None
        prediction = ""
        try:
            records = list(rpred.rpred(model, line, seg))
            prediction = records[0].prediction.strip() if records else ""
        except Exception as exc:  # keep one failed item from hiding the batch
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        gold = gold_by_term.get(term)
        results.append(
            {
                "termo": int(term),
                "arquivo": path.name,
                "predicao": prediction,
                "gold": gold,
                "similaridade": similarity(prediction, gold) if gold else None,
                "tempo_ms": round(elapsed_ms, 1),
                "erro": error,
            }
        )
        print(term, round(elapsed_ms / 1000, 2), prediction or "—", flush=True)

    scored = [r for r in results if r["gold"]]
    exact = sum(normalizar(r["predicao"]) == normalizar(r["gold"]) for r in scored)
    summary = {
        "motor": "kraken",
        "modelo": str(args.model),
        "registros": len(results),
        "avaliados": len(scored),
        "exatos": exact,
        "taxa_exata": exact / len(scored) if scored else 0.0,
        "similaridade_media": sum(r["similaridade"] for r in scored) / len(scored)
        if scored
        else 0.0,
        "tempo_medio_ms": sum(r["tempo_ms"] for r in results) / len(results)
        if results
        else 0.0,
        "resultados": results,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "resultados"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
