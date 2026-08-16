from __future__ import annotations

"""Primeira passada barata de nomes antes do Qwen.

Tesseract/RapidOCR produzem candidatos e confiança. O Qwen fica reservado aos
casos abaixo do limiar, evitando gastar minutos lendo nomes que já estão claros
ou tentando transcrever a página inteira.
"""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from ..imaging.record_regions import recortar_registro
from ..metadata.extractor import extrair_metadados
from ..metadata.normalizer import normalizar_busca


@dataclass
class NomeCandidato:
    valor: str
    confianca: float
    motor: str
    texto_bruto: str
    contexto: str = ""


def recortar_registro_nome(
    image: np.ndarray,
    indice: int,
    total: int,
) -> np.ndarray:
    """Compatibilidade: usa a região centralizada do assento."""
    return recortar_registro(image, indice, total)


class NameCandidateIndexer:
    MOTOR = "ocr-nomes-rapido-v1"
    FONTE = "ocr_nome_rapido"

    def __init__(self, repo, providers, limiar_qwen: float = 0.78) -> None:
        self.repo = repo
        self.providers = list(providers)
        self.limiar_qwen = float(limiar_qwen)

    @staticmethod
    def _candidatos(resultado) -> list[NomeCandidato]:
        return [
            NomeCandidato(
                valor=str(d.valor_tratado).strip(),
                confianca=float(d.confianca),
                motor=d.motor or resultado.motor,
                texto_bruto=resultado.texto_bruto or "",
                contexto=d.contexto or "",
            )
            for d in extrair_metadados(resultado)
            if d.tipo == "nome_registrado" and str(d.valor_tratado).strip()
        ]

    def indexar(self, imagem_id: int, image: np.ndarray, registros: list[dict]) -> dict:
        if not registros or image is None or image.size == 0:
            return {"nomes": [], "incertos": [], "reutilizados": 0}
        nomes: list[dict] = []
        incertos: list[dict] = []
        reutilizados = 0
        total = len(registros)
        for registro in registros:
            existente = self.repo.get_execucao_ocr_ativa(
                imagem_id=imagem_id,
                registro_id=registro["id"],
                motor=self.MOTOR,
            )
            if existente:
                reutilizados += 1
                continue
            recorte = recortar_registro_nome(
                image, int(registro.get("indice_na_imagem") or 0), total
            )
            candidatos: list[NomeCandidato] = []
            inicio = time.perf_counter()
            textos = []
            for provider in self.providers:
                try:
                    if not provider.is_available():
                        continue
                    resultado = provider.recognize(recorte, fast=True)
                    textos.append(resultado.texto_bruto or "")
                    candidatos.extend(self._candidatos(resultado))
                except Exception:
                    continue
            if not candidatos:
                self.repo.criar_execucao_ocr(
                    imagem_id=imagem_id,
                    registro_id=registro["id"],
                    motor=self.MOTOR,
                    texto_bruto="\n".join(textos),
                    tempo_ms=(time.perf_counter() - inicio) * 1000,
                    sucesso=False,
                    erro="nenhum nome candidato",
                )
                incertos.append({"registro_id": registro["id"], "termo": registro.get("termo")})
                continue

            agrupados: dict[str, NomeCandidato] = {}
            for candidato in candidatos:
                chave = " ".join(candidato.valor.casefold().split())
                anterior = agrupados.get(chave)
                if anterior is None or candidato.confianca > anterior.confianca:
                    agrupados[chave] = candidato
            ordenados = sorted(agrupados.values(), key=lambda item: item.confianca, reverse=True)
            melhor = ordenados[0]
            concordantes = sum(
                1 for candidato in candidatos
                if " ".join(candidato.valor.casefold().split())
                == " ".join(melhor.valor.casefold().split())
            )
            confianca = min(1.0, melhor.confianca + (0.12 if concordantes >= 2 else 0.0))
            status = "provavel" if confianca >= self.limiar_qwen else "precisa_revisao"
            contexto = (
                f"Candidatos rápidos: {', '.join(item.valor for item in ordenados[:3])}; "
                f"concordância={concordantes}"
            )
            execucao = self.repo.criar_execucao_ocr(
                imagem_id=imagem_id,
                registro_id=registro["id"],
                motor=self.MOTOR,
                texto_bruto="\n".join(textos),
                tempo_ms=(time.perf_counter() - inicio) * 1000,
                sucesso=True,
            )
            self.repo.salvar_deteccoes_ocr(
                execucao_id=execucao,
                imagem_id=imagem_id,
                registro_id=registro["id"],
                deteccoes=[{
                    "tipo": "nome_registrado",
                    "valor_original": melhor.valor,
                    "valor_tratado": melhor.valor,
                    "valor_normalizado": normalizar_busca(melhor.valor),
                    "confianca": confianca,
                    "motor": melhor.motor,
                    "fonte": self.FONTE,
                    "status": status,
                    "contexto": contexto,
                }],
            )
            item = {
                "registro_id": registro["id"],
                "termo": registro.get("termo"),
                "valor": melhor.valor,
                "confianca": round(confianca, 3),
                "status": status,
            }
            nomes.append(item)
            if status == "precisa_revisao":
                incertos.append(item)

        if incertos and not self.repo.tem_revisao_pendente(imagem_id, "nome_incerto"):
            resumo = "; ".join(
                f"termo {item.get('termo') or '?'}: {item.get('valor', 'sem candidato')}"
                for item in incertos
            )
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="nome_incerto",
                detalhes=(
                    "Nome capturado por OCR rápido abaixo do limiar; "
                    f"enviar somente estes assentos ao Qwen. {resumo}"
                ),
            )
        return {"nomes": nomes, "incertos": incertos, "reutilizados": reutilizados}
