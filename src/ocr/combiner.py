from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from .base import OCRProvider, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class TermoResultado:
    valor: int | None = None
    status: str = "pendente"
    confianca: float = 0.0
    motor_principal: str = ""
    alternativas: list[dict] = field(default_factory=list)
    esperado_min: int | None = None
    esperado_max: int | None = None
    texto_bruto: str = ""


class OCRCombiner:
    def __init__(self, providers: list[OCRProvider] | None = None) -> None:
        self.providers: list[OCRProvider] = providers or []

    def add_provider(self, provider: OCRProvider) -> None:
        self.providers.append(provider)

    def recognize_all(self, image, fast: bool = False) -> list[OCRResult]:
        results = []
        for provider in self.providers:
            if provider.is_available():
                try:
                    results.append(provider.recognize(image, fast=fast))
                except Exception as exc:
                    logger.warning("Provider %s falhou: %s", provider.name, exc)
        return results

    def extrair_termo(
        self,
        results: list[OCRResult],
        esperado_min: int | None = None,
        esperado_max: int | None = None,
        fallback_sequencia: bool = False,
    ) -> TermoResultado:
        """Extrai um termo e, quando solicitado, valida a sequencia cadastrada.

        O OCR continua sendo evidência, não a fonte da ordem. Isso evita que uma
        data ou uma leitura parcial de ``6.801`` faça a sessão saltar de termo.
        """
        todos_textos = " ".join(r.texto_bruto for r in results)
        candidatos: list[dict] = []

        def valor_numerico(raw: str) -> int | None:
            # Nos livros o milhar pode estar separado: 6.801 ou 6 801.
            digitos = re.sub(r"\D", "", raw)
            if not digitos or len(digitos) > 7 or digitos.startswith("0"):
                return None
            valor = int(digitos)
            return valor if 50 <= valor <= 9_999_999 else None

        def resultado_sequencial(
            status: str = "inferido_sequencia", confianca: float = 0.70
        ) -> TermoResultado:
            return TermoResultado(
                valor=esperado_min,
                status=status,
                confianca=confianca,
                motor_principal="sequencia",
                esperado_min=esperado_min,
                esperado_max=esperado_max,
                texto_bruto=todos_textos[:500],
            )

        numero = r"(?:\d{1,3}(?:[.\s]\d{3})|\d{1,7})"
        rotulo = r"(?:termo|numero|n[uú]mero|n[º°])"

        for result in results:
            texto = result.texto_bruto
            is_htr = result.motor == "htr"

            for match in re.finditer(rf"(?i){rotulo}\s*[:.]?\s*({numero})", texto):
                valor = valor_numerico(match.group(1))
                if valor is not None:
                    candidatos.append({
                        "valor": valor,
                        "confianca": 0.95,
                        "motor": result.motor,
                        "tipo_original": "label_numero",
                    })

            tem_rotulo = any(
                c["tipo_original"] == "label_numero" and c["motor"] == result.motor
                for c in candidatos
            )
            if is_htr and not tem_rotulo:
                for match in re.finditer(rf"(?i){rotulo}", texto):
                    trecho = texto[match.start():match.start() + 80]
                    for raw in re.findall(numero, trecho):
                        valor = valor_numerico(raw)
                        if valor is not None:
                            candidatos.append({
                                "valor": valor,
                                "confianca": 0.85,
                                "motor": result.motor,
                                "tipo_original": "post_numero",
                            })
                            break

        # Sem um rótulo, números soltos só entram como evidência fraca. Datas e
        # folhas deixam de vencer um termo que confere com a faixa esperada.
        if not candidatos:
            for result in results:
                for match in re.finditer(rf"(?<!\d)({numero})(?!\d)", result.texto_bruto):
                    valor = valor_numerico(match.group(1))
                    if valor is None:
                        continue
                    is_htr = result.motor == "htr"
                    if is_htr and valor <= 9999:
                        confianca, tipo = 0.55, "numero_generico"
                    elif not is_htr and len(str(valor)) >= 4:
                        confianca, tipo = 0.50, "numero_ocr"
                    else:
                        continue
                    candidatos.append({
                        "valor": valor,
                        "confianca": confianca,
                        "motor": result.motor,
                        "tipo_original": tipo,
                    })

        faixa_estreita = (
            esperado_min is not None
            and esperado_max is not None
            and 0 <= esperado_max - esperado_min <= 20
        )
        if esperado_min is not None and esperado_max is not None:
            for candidato in candidatos:
                if esperado_min <= candidato["valor"] <= esperado_max:
                    candidato["confianca"] += 0.30
                    candidato["confere_sequencia"] = True
                elif faixa_estreita:
                    candidato["confianca"] -= 0.35
                    candidato["confere_sequencia"] = False
                elif abs(candidato["valor"] - esperado_min) < 200:
                    candidato["confianca"] += 0.10

        if not candidatos:
            if fallback_sequencia and esperado_min is not None:
                return resultado_sequencial()
            return TermoResultado(texto_bruto=todos_textos)

        acordo = Counter(c["valor"] for c in candidatos)
        for candidato in candidatos:
            if acordo[candidato["valor"]] >= 2:
                candidato["confianca"] += 0.15

        unicos: dict[int, dict] = {}
        for candidato in candidatos:
            valor = candidato["valor"]
            if valor not in unicos or candidato["confianca"] > unicos[valor]["confianca"]:
                unicos[valor] = candidato
        ordenados = sorted(unicos.values(), key=lambda c: c["confianca"], reverse=True)

        candidatos_na_faixa = [
            c for c in ordenados
            if esperado_min is not None
            and esperado_max is not None
            and esperado_min <= c["valor"] <= esperado_max
        ]
        if faixa_estreita and candidatos_na_faixa:
            ordenados = candidatos_na_faixa + [c for c in ordenados if c not in candidatos_na_faixa]
        elif faixa_estreita and fallback_sequencia:
            melhor_ocr = ordenados[0]
            # Número completo e rotulado pode denunciar uma foto fora de ordem.
            # Leituras parciais como 804 para 6801 não geram falso alerta.
            conflito_forte = (
                melhor_ocr.get("tipo_original") == "label_numero"
                and len(str(melhor_ocr["valor"])) == len(str(esperado_min))
                and melhor_ocr["confianca"] >= 0.60
            )
            inferido = resultado_sequencial(
                "precisa_revisao" if conflito_forte else "inferido_sequencia",
                0.40 if conflito_forte else 0.70,
            )
            inferido.alternativas = ordenados[:5]
            return inferido

        melhor = ordenados[0]
        confianca = max(0.0, min(1.0, melhor["confianca"]))
        if confianca >= 0.95:
            status = "confirmado"
        elif confianca >= 0.80:
            status = "provavel"
        elif confianca >= 0.60:
            status = "duvidoso"
        else:
            status = "precisa_revisao"

        return TermoResultado(
            valor=melhor["valor"],
            status=status,
            confianca=confianca,
            motor_principal=melhor["motor"],
            alternativas=ordenados[:5],
            esperado_min=esperado_min,
            esperado_max=esperado_max,
            texto_bruto=todos_textos[:500],
        )

    def extrair_folha(self, results: list[OCRResult]) -> TermoResultado:
        candidatos: list[dict] = []
        for result in results:
            for match in re.finditer(
                r"(?i)(?:folha|fls?\.?|f[º°])\s*[:.]?\s*(\d{1,4})",
                result.texto_bruto,
            ):
                candidatos.append({
                    "valor": int(match.group(1)),
                    "confianca": 0.8,
                    "motor": result.motor,
                })
        if not candidatos:
            return TermoResultado()
        candidatos.sort(key=lambda c: c["confianca"], reverse=True)
        melhor = candidatos[0]
        return TermoResultado(
            valor=melhor["valor"],
            status="identificado" if melhor["confianca"] >= 0.7 else "duvidoso",
            confianca=melhor["confianca"],
            motor_principal=melhor["motor"],
            alternativas=candidatos[:3],
        )

    def estimar_registros_por_face(self, results: list[OCRResult]) -> int:
        valores: list[int] = []
        for result in results:
            for match in re.finditer(
                r"(?i)(?:termo|numero|n[uú]mero|n[º°])\s*[:.]?\s*(\d+)",
                result.texto_bruto,
            ):
                valor = int(match.group(1))
                if 1 <= valor <= 9999:
                    valores.append(valor)
        if len(valores) >= 2:
            valores = sorted(set(valores))
            diferencas = [
                b - a for a, b in zip(valores, valores[1:]) if 1 <= b - a <= 100
            ]
            if diferencas:
                return Counter(diferencas).most_common(1)[0][0]
        return 1


def texto_evidencia_termo(registro: dict) -> str:
    """Texto curto mostrando se o OCR leu mesmo o numero do termo.

    A sequencia auditada continua sendo a decisao; esta funcao apenas expoe a
    evidencia (ou a falta dela) para o operador na Consulta e no Revisor.
    """
    status = str(registro.get("termo_status") or "").strip().lower()
    ocr = str(registro.get("ocr_termo") or "").strip()
    confianca = float(registro.get("confianca_termo") or 0)
    sequencia = registro.get("termo")
    if status in ("confirmado", "provavel"):
        if ocr:
            pct = f" ({confianca * 100:.0f}%)" if confianca else ""
            return f"OCR leu o termo manuscrito: {ocr}{pct} ✓"
        return "OCR do termo: leitura confirmada ✓"
    if status in ("duvidoso", "precisa_revisao"):
        if ocr and str(sequencia) not in (None, "", "None") and ocr != str(sequencia):
            return f"OCR leu {ocr}, mas a sequência diz {sequencia} ✗ — confira a foto"
        if ocr:
            return f"OCR leu {ocr} com dúvida — confira a foto"
        return "OCR do termo: leitura duvidosa — confira a foto"
    if status == "nao_identificado":
        return "OCR do termo: número não identificado"
    if status == "inferido_sequencia":
        if ocr:
            return f"Termo pela sequência (OCR alternativo: {ocr})"
        return "Termo pela sequência; OCR não leu o número manuscrito"
    if ocr:
        return f"OCR do termo: {ocr}"
    return ""
