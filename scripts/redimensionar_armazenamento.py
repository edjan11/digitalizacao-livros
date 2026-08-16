"""Gera derivadas 300 DPI para armazenamento sem alterar os originais."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imaging.storage_derivative import criar_derivada_armazenamento


EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def executar(entrada: Path, saida: Path, target_dpi: float, quality: int) -> dict:
    if entrada.is_file():
        paginas = [entrada]
    else:
        paginas = sorted(
            path for path in entrada.iterdir()
            if path.is_file() and path.suffix.lower() in EXTS
        )
    if not paginas:
        raise RuntimeError("Nenhuma imagem encontrada")
    if entrada.resolve() == saida.resolve():
        raise ValueError("A pasta de saida precisa ser separada da entrada")

    inicio = time.perf_counter()
    resultados = []
    erros = []
    for pagina in paginas:
        destino = saida / f"{pagina.stem}.jpg"
        try:
            resultados.append(
                criar_derivada_armazenamento(
                    pagina,
                    destino,
                    target_dpi=target_dpi,
                    jpeg_quality=quality,
                    jpeg_subsampling=2,
                ).to_dict()
            )
        except Exception as exc:  # a fila continua e o erro fica auditavel
            erros.append({"source_path": str(pagina), "error": str(exc)})

    source_bytes = sum(item["source_bytes"] for item in resultados)
    output_bytes = sum(item["output_bytes"] for item in resultados)
    report = {
        "version": 1,
        "entrada": str(entrada),
        "saida": str(saida),
        "target_dpi": target_dpi,
        "jpeg_quality": quality,
        "total_entrada": len(paginas),
        "total_processado": len(resultados),
        "total_erros": len(erros),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "reduction_percent": round((1 - output_bytes / source_bytes) * 100, 2) if source_bytes else 0,
        "elapsed_seconds": round(time.perf_counter() - inicio, 3),
        "items": resultados,
        "errors": erros,
        "quality_policy": (
            "Lanczos, JPEG 4:2:0, qualidade indicada, DPI escrito em 300; "
            "originais permanecem imutaveis."
        ),
    }
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "storage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--dpi", type=float, default=300)
    parser.add_argument("--quality", type=int, default=75)
    args = parser.parse_args()
    try:
        report = executar(args.entrada, args.saida, args.dpi, args.quality)
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "total_entrada": report["total_entrada"],
        "total_processado": report["total_processado"],
        "total_erros": report["total_erros"],
        "reduction_percent": report["reduction_percent"],
        "saida": report["saida"],
        "report": str(args.saida / "storage_report.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
