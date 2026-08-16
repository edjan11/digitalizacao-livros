"""Analise completa de imagens - OCR rapido, sequencia de termos, registros por face."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analisar_pasta(pasta: str, esperado_min: int = None, esperado_max: int = None, fast: bool = True) -> dict:
    from src.ocr.engines import TesseractProvider, RapidOCRProvider
    from src.ocr.combiner import OCRCombiner

    pasta_path = Path(pasta)
    if not pasta_path.exists():
        logger.error("Pasta nao encontrada: %s", pasta)
        return {}

    imagens = sorted([f for f in pasta_path.iterdir()
                      if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
                      and '_prep' not in f.name and '_thumb' not in f.name])

    if not imagens:
        logger.error("Nenhuma imagem encontrada")
        return {}

    logger.info("=" * 70)
    logger.info("ANALISE: %s (%d imagens)", pasta_path.name, len(imagens))
    if fast:
        logger.info("Modo: RAPIDO (imagens redimensionadas para velocidade)")
    logger.info("=" * 70)

    combiner = OCRCombiner()
    tess = TesseractProvider()
    rapid = RapidOCRProvider()
    if tess.is_available():
        combiner.add_provider(tess)
    if rapid.is_available():
        combiner.add_provider(rapid)

    resultados = []
    termos_seq = []
    folhas_seq = []

    for idx, img_path in enumerate(imagens):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        results = combiner.recognize_all(image, fast=fast)
        termo = combiner.extrair_termo(results, esperado_min, esperado_max)
        folha = combiner.extrair_folha(results)

        tempos = {r.motor: round(r.tempo_ms, 1) for r in results}

        arquivo = img_path.name
        t_val = termo.valor
        t_st = termo.status
        f_val = folha.valor

        if t_val is not None:
            termos_seq.append({"arquivo": arquivo, "termo": t_val, "status": t_st, "motor": termo.motor_principal})
        if f_val is not None:
            folhas_seq.append({"arquivo": arquivo, "folha": f_val, "status": folha.status})

        logger.info("[%3d/%d] %-40s | termo=%s (%s, %.2f) | folha=%s | %s",
                     idx + 1, len(imagens), arquivo[:40],
                     str(t_val) if t_val else '?',
                     t_st[:4],
                     termo.confianca,
                     str(f_val) if f_val else '?',
                     " ".join(f"{m}={t}ms" for m, t in tempos.items()))

    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS: %s", pasta_path.name)
    logger.info("=" * 70)

    if termos_seq:
        logger.info("\nSEQUENCIA DE TERMOS:")
        regs_por_face = _calcular_registros_por_face(termos_seq)
        logger.info("Registros por face estimados: %d", regs_por_face)

        logger.info("\n%-6s %-40s %-8s %-8s %s", "Ordem", "Arquivo", "Termo", "Diff", "Status")
        logger.info("-" * 85)

        ultimo_termo = None
        anomalias = []
        for i, t in enumerate(termos_seq):
            diff = ""
            if ultimo_termo is not None and t["termo"] is not None:
                d = t["termo"] - ultimo_termo
                diff = f"+{d}"
                if d != regs_por_face and d > 0:
                    marker = " !!"
                    anomalias.append((ultimo_termo, t["termo"], d, regs_por_face, t["arquivo"]))
                else:
                    marker = ""
            else:
                diff = "--"
                marker = ""
            logger.info("%-6d %-40s %-8s %-8s %s%s",
                         i + 1, t["arquivo"][:40], str(t["termo"]), diff, t["status"][:8], marker)
            ultimo_termo = t["termo"]

        if anomalias:
            logger.info("\nANOMALIAS NA SEQUENCIA:")
            for t1, t2, diff, esperado, arq in anomalias:
                logger.info("  Termo %s -> %s (diff=%d, esperado=%d) em %s",
                           t1, t2, diff, esperado, arq)

    if folhas_seq:
        logger.info("\nFOLHAS DETECTADAS: %d", len(folhas_seq))
        for f in folhas_seq[:5]:
            logger.info("  %s: Folha %s (%s)", f["arquivo"][:40], f["folha"], f["status"])

    tempos_total = sum(
        sum(r.tempo_ms for r in combiner.recognize_all(cv2.imread(str(img)), fast=fast)
            if hasattr(r, 'tempo_ms'))
        for img in imagens[:1]
    ) if imagens else 0

    logger.info("\nTempo medio por imagem: ~%.0fms", tempos_total if tempos_total > 0 else 0)

    return {
        "total": len(imagens),
        "termos_encontrados": len(termos_seq),
        "folhas_encontradas": len(folhas_seq),
        "registros_por_face": regs_por_face if termos_seq else 1,
        "anomalias": len(anomalias) if termos_seq else 0,
        "termos_seq": termos_seq,
        "folhas_seq": folhas_seq,
    }


def _calcular_registros_por_face(termos_seq: list[dict]) -> int:
    valores = [t["termo"] for t in termos_seq if t["termo"] is not None]
    if len(valores) < 2:
        return 1
    diffs = []
    for i in range(len(valores) - 1):
        diff = valores[i + 1] - valores[i]
        if 1 <= diff <= 100:
            diffs.append(diff)
    if not diffs:
        return 1
    return Counter(diffs).most_common(1)[0][0]


if __name__ == "__main__":
    import os

    todas_pastas = {}

    base = r"D:\Users\Usuario\Desktop\teste"
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full):
            for sub in os.listdir(full):
                sub_full = os.path.join(full, sub)
                if sub.endswith("_pages") and os.path.isdir(sub_full):
                    jpgs = [f for f in os.listdir(sub_full) if f.endswith('.jpg') and '_prep' not in f]
                    if jpgs:
                        todas_pastas[sub] = sub_full
            if entry.endswith("_pages") and os.path.isdir(full):
                jpgs = [f for f in os.listdir(full) if f.endswith('.jpg') and '_prep' not in f]
                if jpgs:
                    todas_pastas[entry] = full

    if not todas_pastas:
        logger.error("Nenhuma pasta _pages encontrada em %s", base)
        sys.exit(1)

    logger.info("Pastas encontradas:")
    for nome, path in todas_pastas.items():
        jpgs = [f for f in os.listdir(path) if f.endswith('.jpg') and '_prep' not in f]
        logger.info("  %s: %s (%d imagens)", nome, path, len(jpgs))

    for nome, path in sorted(todas_pastas.items()):
        try:
            analisar_pasta(path, fast=True)
            logger.info("\n")
        except Exception as e:
            logger.error("Erro em %s: %s", nome, e)
