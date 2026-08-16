"""Varre todas as fotos reais do livro e mede a retificacao usada na exibicao.

Nao altera nenhum arquivo: apenas aplica ``retificar_formulario`` em memoria
(a mesma funcao usada pelo revisor/miniatura) e reporta quantas fotos seriam
exibidas endireitadas com a confianca exigida (>= 0.50).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imaging.document import retificar_formulario


def _processar(caminho: Path) -> dict:
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        return {"arquivo": str(caminho), "erro": "nao decodificou"}
    altura, largura = imagem.shape[:2]
    inicio = time.perf_counter()
    resultado = retificar_formulario(imagem)
    gasto = time.perf_counter() - inicio
    # A exibicao aceita qualquer retificacao aplicada: applied=True garante ao
    # menos o alinhamento horizontal; applied=False devolve a copia sem mudanca.
    return {
        "arquivo": str(caminho),
        "pixels": altura * largura,
        "applied": bool(resultado.applied),
        "confidence": round(float(resultado.confidence), 3),
        "angulo": round(float(resultado.angle_degrees), 3),
        "motivo": resultado.reason,
        "exibir_retificada": bool(resultado.applied),
        "tempo_ms": round(gasto * 1000, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=r"D:\A - 07")
    ap.add_argument("--saida", default=".tmp_retificacao_livro.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    fotos = sorted(Path(args.raiz).rglob("*.jpg"))
    if not fotos:
        raise SystemExit("nenhuma foto encontrada")

    inicio_lote = time.perf_counter()
    resultados = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for item in executor.map(_processar, fotos):
            resultados.append(item)
            if len(resultados) % 50 == 0:
                print(f"{len(resultados)}/{len(fotos)}...", flush=True)

    total = len(resultados)
    com_erro = [r for r in resultados if r.get("erro")]
    validos = [r for r in resultados if not r.get("erro")]
    exibem_retificada = [r for r in validos if r["exibir_retificada"]]
    aplicadas = [r for r in validos if r["applied"]]
    nao_aplicadas = [r for r in validos if not r["applied"]]

    confiancas = [float(r["confidence"]) for r in validos]
    tempos = [float(r["tempo_ms"]) for r in validos]
    resumo = {
        "raiz": str(Path(args.raiz)),
        "total": total,
        "erros": len(com_erro),
        "validas": len(validos),
        "exibem_retificada": len(exibem_retificada),
        "taxa_exibem_retificada": round(len(exibem_retificada) / len(validos), 4) if validos else 0.0,
        "aplicadas_mas_abaixo_limiar": len(aplicadas) - len(exibem_retificada),
        "nao_aplicadas": len(nao_aplicadas),
        "confianca_media": round(sum(confiancas) / len(confiancas), 3) if confiancas else 0.0,
        "confianca_min": round(min(confiancas), 3) if confiancas else 0.0,
        "confianca_max": round(max(confiancas), 3) if confiancas else 0.0,
        "tempo_medio_ms": round(sum(tempos) / len(tempos), 1) if tempos else 0.0,
        "tempo_total_s": round(time.perf_counter() - inicio_lote, 1),
        "motivos": {
            motivo: sum(1 for r in nao_aplicadas if r["motivo"] == motivo)
            for motivo in sorted({r["motivo"] for r in nao_aplicadas})
        },
        "resultados": resultados,
    }
    Path(args.saida).write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: resumo[k] for k in resumo if k != "resultados"}, ensure_ascii=False, indent=1
    ))


if __name__ == "__main__":
    main()
