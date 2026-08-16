"""Teste real com imagens de livros - OCR, HTR, termos, folhas, registros por face."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import re
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extrair_tokens_avancado(texto: str) -> list[dict]:
    """Extrai termos, folhas e outros tokens com regex melhorado para livros."""
    tokens = []

    for m in re.finditer(r'(?i)(?:termo|n[º°])\s*[:.]?\s*(\d{1,7})', texto):
        tokens.append({"tipo": "termo", "valor": m.group(1), "source": "label"})

    for m in re.finditer(r'(?i)(?:folha|fls?\.?|f[º°])\s*[:.]?\s*(\d{1,4})\s*(v|verso|frente)?', texto):
        tokens.append({"tipo": "folha", "valor": m.group(1), "face": m.group(2) or "", "source": "label"})

    for m in re.finditer(r'(?i)(?:pag|p[áa]g)\s*[:.]?\s*(\d{1,4})', texto):
        tokens.append({"tipo": "pagina", "valor": m.group(1), "source": "label"})

    for m in re.finditer(r'(?i)(?:livro|lv\.?)\s*[:.]?\s*([A-Za-z0-9\-]+)', texto):
        tokens.append({"tipo": "livro", "valor": m.group(1), "source": "label"})

    for m in re.finditer(r'\b(19\d{2}|20\d{2})\b', texto):
        tokens.append({"tipo": "ano", "valor": m.group(1), "source": "ano"})

    for m in re.finditer(r'\b(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{2,4})\b', texto):
        tokens.append({"tipo": "data", "valor": f"{m.group(1)}/{m.group(2)}/{m.group(3)}", "source": "data"})

    return tokens


def detectar_registros_por_face(textos_ocr: list[str]) -> int:
    """Tenta detectar quantos registros existem por face analisando padroes de termos."""
    todos_termos = []
    for texto in textos_ocr:
        for m in re.finditer(r'(?i)(?:termo|n[º°])\s*[:.]?\s*(\d+)', texto):
            todos_termos.append(int(m.group(1)))

    if len(todos_termos) < 2:
        for texto in textos_ocr:
            numeros = [int(m.group(1)) for m in re.finditer(r'\b(\d{4,7})\b', texto)]
            if len(numeros) >= 2:
                todos_termos = numeros
                break

    if len(todos_termos) >= 2:
        diferencas = []
        for i in range(len(todos_termos) - 1):
            diff = todos_termos[i+1] - todos_termos[i]
            if 1 <= diff <= 5:
                diferencas.append(diff)
        if diferencas:
            contagem = Counter(diferencas)
            mais_comum = contagem.most_common(1)[0][0]
            return mais_comum

    return 1


def processar_pasta_imagens(pasta: str) -> dict:
    """Processa todas as imagens de uma pasta e retorna analise completa."""
    from src.ocr.engines import TesseractProvider, RapidOCRProvider
    from src.ocr.htr_engine import HTREngine

    pasta_path = Path(pasta)
    if not pasta_path.exists():
        logger.error("Pasta nao encontrada: %s", pasta)
        return {}

    imagens = sorted(
        list(pasta_path.glob("*.jpg")) +
        list(pasta_path.glob("*.jpeg")) +
        list(pasta_path.glob("*.png"))
    )

    if not imagens:
        logger.error("Nenhuma imagem encontrada")
        return {}

    logger.info("=" * 60)
    logger.info("PROCESSANDO %d IMAGENS: %s", len(imagens), pasta_path.name)
    logger.info("=" * 60)

    tess = TesseractProvider()
    rapid = RapidOCRProvider()
    htr = HTREngine()

    resultados = []
    todos_textos = []
    termos_encontrados = []

    for idx, img_path in enumerate(imagens):
        logger.info("\n[%d/%d] %s", idx + 1, len(imagens), img_path.name)
        image = cv2.imread(str(img_path))

        if image is None:
            logger.warning("  ERRO: nao foi possivel ler")
            continue

        h, w = image.shape[:2]
        logger.info("  Dimensoes: %dx%d", w, h)

        result = {"arquivo": img_path.name, "path": str(img_path), "dimensoes": f"{w}x{h}", "engines": {}}

        if tess.is_available():
            t0 = cv2.getTickCount()
            r = tess.recognize(image)
            t = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
            tokens = extrair_tokens_avancado(r.texto_bruto)
            result["engines"]["tesseract"] = {
                "texto": r.texto_bruto[:300], "tempo_ms": round(t, 1), "tokens": tokens
            }
            todos_textos.append(r.texto_bruto)
            for tok in tokens:
                if tok["tipo"] == "termo":
                    termos_encontrados.append((img_path.name, "tesseract", int(tok["valor"])))
            if r.texto_bruto.strip():
                logger.info("  Tesseract (%dms): %s", round(t, 1), r.texto_bruto.strip()[:120])

        if rapid.is_available():
            t0 = cv2.getTickCount()
            r = rapid.recognize(image)
            t = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
            tokens = extrair_tokens_avancado(r.texto_bruto)
            result["engines"]["rapidocr"] = {
                "texto": r.texto_bruto[:300], "tempo_ms": round(t, 1), "tokens": tokens
            }
            todos_textos.append(r.texto_bruto)
            for tok in tokens:
                if tok["tipo"] == "termo":
                    termos_encontrados.append((img_path.name, "rapidocr", int(tok["valor"])))
            if r.texto_bruto.strip():
                logger.info("  RapidOCR (%dms): %s", round(t, 1), r.texto_bruto.strip()[:120])

        if htr.is_available():
            logger.info("  HTR: testando...")
            try:
                t0 = cv2.getTickCount()
                r = htr.recognize(image)
                t = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
                tokens = extrair_tokens_avancado(r.texto_bruto)
                result["engines"]["htr"] = {
                    "texto": r.texto_bruto[:300], "tempo_ms": round(t, 1), "tokens": tokens
                }
                todos_textos.append(r.texto_bruto)
                for tok in tokens:
                    if tok["tipo"] == "termo":
                        termos_encontrados.append((img_path.name, "htr", int(tok["valor"])))
                if r.texto_bruto.strip():
                    logger.info("  HTR (%dms): %s", round(t, 1), r.texto_bruto.strip()[:120])
            except Exception as e:
                logger.warning("  HTR indisponivel: %s", str(e)[:80])

        resultados.append(result)

    logger.info("\n" + "=" * 60)
    logger.info("RESUMO DA ANALISE")
    logger.info("=" * 60)

    regs_por_face = detectar_registros_por_face(todos_textos)
    logger.info("Registros por face estimados: %d", regs_por_face)

    if termos_encontrados:
        logger.info("\nTermos encontrados (ordenados):")
        termos_encontrados.sort(key=lambda x: x[2])
        for arquivo, motor, termo in termos_encontrados:
            logger.info("  [%s] %s -> Termo %d", motor, arquivo, termo)

        termo_vals = [t[2] for t in termos_encontrados]
        if len(termo_vals) >= 2:
            saltos = []
            for i in range(len(termo_vals) - 1):
                diff = termo_vals[i+1] - termo_vals[i]
                if diff != regs_por_face and diff > 0:
                    saltos.append((termo_vals[i], termo_vals[i+1], diff))

            if saltos:
                logger.info("\nPossiveis irregularidades na sequencia:")
                for t1, t2, diff in saltos:
                    esperado = t1 + regs_por_face
                    logger.info("  Termo %d -> %d (diff=%d, esperado=%d) %s",
                               t1, t2, diff, esperado,
                               "OK" if diff == regs_por_face else "ATENCAO")

    logger.info("\nArquivos processados: %d", len(resultados))
    engines_ativas = set()
    for r in resultados:
        engines_ativas.update(r["engines"].keys())
    logger.info("Motores ativos: %s", ", ".join(engines_ativas) if engines_ativas else "nenhum")

    return {
        "total_imagens": len(resultados),
        "resultados": resultados,
        "registros_por_face": regs_por_face,
        "termos_encontrados": [(a, m, t) for a, m, t in termos_encontrados],
    }


if __name__ == "__main__":
    import os

    pastas_teste = [
        r"D:\Users\Usuario\Desktop\teste\teste1_pages",
        r"D:\Users\Usuario\Desktop\teste\teste_pages",
        r"D:\Users\Usuario\Desktop\teste\teste 3\teste1_pages",
        r"D:\Users\Usuario\Desktop\teste\TESTE 4\doc_05082026_113657_031524_pages",
        r"D:\Users\Usuario\Desktop\teste\TESTE 5\doc_07082026_095318_031805_pages",
        r"D:\Users\Usuario\Desktop\teste\teste 6\doc_07082026_103228_031862_pages",
    ]

    pasta_encontrada = None
    for p in pastas_teste:
        if os.path.isdir(p):
            pasta_encontrada = p
            break

    if pasta_encontrada:
        resultado = processar_pasta_imagens(pasta_encontrada)
    else:
        print("Nenhuma pasta de teste encontrada.")
        print("Pastas procuradas:")
        for p in pastas_teste:
            print(f"  {p}")
