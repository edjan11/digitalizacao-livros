"""Benchmark do GOT-OCR 2.0 nos mesmos 20 recortes de nome do A-07.

Comparacao justa com o Qwen/Kraken: mesma pasta de recortes (.tmp_qwen_20_line),
mesmo gabarito e mesmas metricas de Levenshtein. O GOT e um OCR generativo de
pagina inteira, entao o texto impresso do formulario pode aparecer na resposta;
removemos apenas o rotulo fixo "que recebeu o nome de" para nao penalizar o
modelo por transcrever a tinta impressa (o Qwen recebe essa instrucao no
prompt; o GOT nao recebe prompt nesta modalidade).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.got_ocr_engine import GOTOCRProvider, _texto_repetitivo


def normalizar(texto: str | None) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9]+", "", texto.lower())
    return texto


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


def limpar_nome(got_texto: str) -> str:
    """Remove o rotulo impresso fixo que o GOT transcreve junto com a linha."""
    texto = " ".join(got_texto.replace("\n", " ").split())
    padroes = (
        r"recebeu o nome de",
        r"recebeu o nome",
        r"que recebeu",
        r"o nome de",
        r"nome de",
    )
    for padrao in padroes:
        pos = re.search(padrao, texto, flags=re.IGNORECASE)
        if pos:
            texto = texto[pos.end():]
            break
    return texto.strip(" :.-–—;,")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".tmp_qwen_20_line")
    ap.add_argument("--model", default="models/got-ocr-2.0-hf")
    ap.add_argument("--gold", default="scripts/gabarito_qwen_a07_20.json")
    ap.add_argument("--output", default=".tmp_got_20_result.json")
    args = ap.parse_args()

    gold_data = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    gold_by_term = {str(k): v for k, v in gold_data.items()}

    provider = GOTOCRProvider(model_path=args.model, permitir_download=False)
    provider.load()
    results = []
    files = sorted(
        (p for p in Path(args.input).glob("*.jpg") if p.name != "montagem.jpg"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for path in files:
        term = path.stem.split("_")[-1]
        started = time.perf_counter()
        error = None
        prediction = ""
        try:
            resultado = provider.recognize(path, fast=False)
            bruto = resultado.texto_bruto.strip()
            if _texto_repetitivo(bruto):
                raise RuntimeError("GOT-OCR gerou resposta repetitiva")
            prediction = limpar_nome(bruto)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        gold = gold_by_term.get(term)
        results.append({
            "termo": int(term),
            "arquivo": path.name,
            "predicao": prediction,
            "gold": gold,
            "similaridade": similarity(prediction, gold) if gold else None,
            "tempo_ms": round(elapsed_ms, 1),
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
        "motor": "got-ocr-2.0",
        "modelo": str(args.model),
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
