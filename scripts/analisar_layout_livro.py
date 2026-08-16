"""Calibra layouts de um conjunto de faces sem chamar o Qwen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imaging.adaptive_layout import AdaptiveLayoutDetector, diagnosticos_da_observacao


EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    if count <= 1:
        return [0]
    valores = np.linspace(0, total - 1, count)
    return list(dict.fromkeys(int(round(float(valor))) for valor in valores))


def _paginas(entrada: Path) -> list[Path]:
    if entrada.is_file():
        return [entrada] if entrada.suffix.lower() in EXTS else []
    return sorted(path for path in entrada.iterdir() if path.is_file() and path.suffix.lower() in EXTS)


def analisar(entrada: Path, saida: Path, bootstrap: int, validacao: int) -> dict:
    paginas = _paginas(entrada)
    if not paginas:
        raise RuntimeError("Nenhuma imagem encontrada na entrada")
    saida.mkdir(parents=True, exist_ok=True)
    detector = AdaptiveLayoutDetector(saida / "layout_templates.json")
    iniciais = list(range(min(bootstrap, len(paginas))))
    distribuidas = _indices(len(paginas), validacao)
    escolhidas = list(dict.fromkeys(iniciais + distribuidas))
    observacoes = []
    artefatos = []
    for index in escolhidas:
        caminho = paginas[index]
        image = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        if image is None:
            continue
        obs = detector.classificar(image, page_number=index + 1)
        observacoes.append({"pagina": index + 1, "arquivo": str(caminho), **obs.to_dict()})
        artefatos.append({"pagina": index + 1, "arquivo": str(caminho), **diagnosticos_da_observacao(image, obs, saida, f"pagina_{index + 1:04d}")})
    report = {
        "version": 1,
        "entrada": str(entrada),
        "total_paginas": len(paginas),
        "paginas_bootstrap": [index + 1 for index in iniciais],
        "paginas_validacao": [index + 1 for index in distribuidas],
        "layouts_encontrados": len(detector.store.templates),
        "templates": [template.to_dict() for template in detector.store.templates],
        "observacoes": observacoes,
        "artefatos": artefatos,
        "qwen": {
            "chamado": False,
            "estrategia": "OpenCV classifica o layout; Qwen recebe apenas crops de nome/termo",
            "benchmark_pendente": "medir pagina inteira, crops individuais e batches 1/4/8/16",
        },
        "validacao_pendente": [
            "confirmar visualmente todos os crops gerados",
            "validar transicoes entre layouts no livro completo",
            "medir qualidade e tempo do Qwen nos crops",
        ],
    }
    (saida / "layout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entrada", type=Path, help="Pasta ou imagem de faces do livro")
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=5)
    parser.add_argument("--validacao", type=int, default=5)
    args = parser.parse_args()
    if args.saida:
        saida = args.saida
    elif args.entrada.is_dir():
        saida = args.entrada / "_layout_analysis"
    else:
        saida = args.entrada.parent / "_layout_analysis"
    try:
        report = analisar(args.entrada, saida, max(1, args.bootstrap), max(1, args.validacao))
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "layouts_encontrados": report["layouts_encontrados"],
        "paginas_analisadas": len(report["observacoes"]),
        "report": str(saida / "layout_report.json"),
        "debug": str(saida / "debug"),
        "crops": str(saida / "crops"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
