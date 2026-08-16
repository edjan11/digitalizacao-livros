from __future__ import annotations

"""Benchmark reproducible do nome manuscrito em 20 registros do A-07.

O script usa exatamente os recortes produzidos por ``amostra_qwen.py``.  O
modelo fica carregado uma única vez, como no trabalhador de produção.  Depois
da execução, um arquivo ``--gold`` pode ser informado para calcular a distância
de Levenshtein sem misturar a transcrição humana com a saída do modelo.
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.qwen_vl_engine import QwenAreaAnalyzer, preparar_imagem_qwen


def normalizar(texto: str | None) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"[^a-z0-9]+", " ", texto.lower())
    return " ".join(texto.split())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pasta", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    gold: dict[str, str] = {}
    if args.gold and args.gold.exists():
        gold = json.loads(args.gold.read_text(encoding="utf-8"))

    fotos = sorted(
        (p for p in args.pasta.glob("*.jpg") if p.name != "montagem.jpg"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not fotos:
        raise SystemExit("nenhum recorte encontrado")

    inicio_lote = time.perf_counter()
    analisador = QwenAreaAnalyzer(
        max_new_tokens=32,
        min_pixels=128 * 28 * 28,
        max_pixels=args.max_pixels,
        dtype=args.dtype,
        threads=args.threads,
    )
    instrucao = (
        "A imagem e um recorte de um registro civil brasileiro. "
        "Leia SOMENTE o nome manuscrito que vem depois das palavras "
        "impressas 'que recebeu o nome de'. Ignore todo o resto. "
        "Responda apenas com o nome completo, sem explicacao. "
        "Se uma letra estiver ilegive, preserve a duvida e nao invente."
    )
    resultados: list[dict[str, object]] = []
    for foto in fotos:
        termo = foto.stem.split("_")[-1]
        ini = time.perf_counter()
        resultado = analisador.analisar(
            preparar_imagem_qwen(foto),
            instrucao=instrucao,
            tipo="nome_registrado",
        )
        pred = (resultado.texto_bruto or "").strip()
        item: dict[str, object] = {
            "termo": int(termo),
            "foto": foto.name,
            "predicao": pred,
            "tempo_ms": round((time.perf_counter() - ini) * 1000),
            "tempo_modelo_ms": round(resultado.tempo_ms),
        }
        esperado = gold.get(str(termo))
        if esperado:
            a, b = normalizar(esperado), normalizar(pred)
            distancia = levenshtein(a, b)
            item.update({
                "verdade_termo": esperado,
                "distancia_levenshtein": distancia,
                "similaridade": round(
                    1.0 - distancia / max(len(a), len(b), 1), 4
                ),
                "exato_normalizado": a == b,
            })
        resultados.append(item)
        print(
            f"{termo}: {pred} | {item['tempo_ms']} ms",
            flush=True,
        )

    resumo: dict[str, object] = {
        "configuracao": {
            "quantidade": len(resultados),
            "max_pixels": args.max_pixels,
            "threads": args.threads,
            "dtype": args.dtype,
            "recorte": "bbox_linha_nome (registro sem averbação)",
        },
        "tempo_total_ms": round((time.perf_counter() - inicio_lote) * 1000),
        "resultados": resultados,
    }
    avaliados = [r for r in resultados if "similaridade" in r]
    if avaliados:
        resumo["metrica"] = {
            "avaliados": len(avaliados),
            "exatos": sum(bool(r["exato_normalizado"]) for r in avaliados),
            "taxa_exata": round(
                sum(bool(r["exato_normalizado"]) for r in avaliados)
                / len(avaliados), 4
            ),
            "similaridade_media": round(
                sum(float(r["similaridade"]) for r in avaliados)
                / len(avaliados), 4
            ),
            "distancia_media": round(
                sum(int(r["distancia_levenshtein"]) for r in avaliados)
                / len(avaliados), 2
            ),
            "tempo_medio_ms": round(
                sum(int(r["tempo_ms"]) for r in avaliados) / len(avaliados)
            ),
        }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULTADO={args.saida}")


if __name__ == "__main__":
    main()
