from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", texto.lower()).split())


def distancia(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(atual[-1] + 1, anterior[j] + 1,
                             anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultado", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    resultado = json.loads(args.resultado.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    avaliados = []
    for item in resultado.get("resultados", []):
        esperado = gold.get(str(item["termo"]))
        if not esperado:
            item["gabarito_status"] = "sem_gabarito_seguro"
            continue
        a, b = normalizar(esperado), normalizar(str(item.get("predicao", "")))
        d = distancia(a, b)
        item.update({
            "verdade_termo": esperado,
            "distancia_levenshtein": d,
            "similaridade": round(1 - d / max(len(a), len(b), 1), 4),
            "exato_normalizado": a == b,
            "gabarito_status": "avaliado",
        })
        avaliados.append(item)
    exatos = sum(bool(item["exato_normalizado"]) for item in avaliados)
    resumo = resultado.setdefault("metrica", {})
    resumo.update({
        "processados": len(resultado.get("resultados", [])),
        "avaliados_com_gabarito": len(avaliados),
        "sem_gabarito_seguro": len(resultado.get("resultados", [])) - len(avaliados),
        "exatos_normalizados": exatos,
        "taxa_exata": round(exatos / len(avaliados), 4) if avaliados else 0,
        "similaridade_media": round(
            sum(float(item["similaridade"]) for item in avaliados) / len(avaliados), 4
        ) if avaliados else 0,
        "distancia_media": round(
            sum(int(item["distancia_levenshtein"]) for item in avaliados) / len(avaliados), 2
        ) if avaliados else 0,
        "tempo_medio_ms": round(
            sum(int(item["tempo_ms"]) for item in avaliados) / len(avaliados)
        ) if avaliados else 0,
    })
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"RESULTADO={args.saida}")


if __name__ == "__main__":
    main()
