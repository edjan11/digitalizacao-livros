from __future__ import annotations

from dataclasses import dataclass
import json
import re

from ..ocr.base import OCRResult
from .normalizer import normalizar_busca, tratar_valor


@dataclass
class MetadataDetection:
    tipo: str
    valor_original: str
    valor_tratado: str
    valor_normalizado: str
    confianca: float
    motor: str
    fonte: str = "ocr"
    status: str = "detectado"
    bbox_json: str | None = None
    contexto: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_PARADAS_NOME = re.compile(
    r"(?i)\b(?:filh[oa]|nasc(?:eu|ida|ido)?|sexo|brasileir[oa]|natural|"
    r"domiciliad[oa]|s[aã]o av[oó]s|em cart[oó]rio|perante|declaro(?:u)?)\b"
)


def _limpar_nome(valor: str) -> str:
    nome = tratar_valor(valor)
    nome = _PARADAS_NOME.split(nome, maxsplit=1)[0]
    nome = re.sub(r"^[^A-Za-zÀ-ÿ]+", "", nome)
    nome = re.sub(r"[^A-Za-zÀ-ÿ'´`\- ]+$", "", nome)
    nome = re.sub(r"\s+", " ", nome).strip()
    return nome[:100]


def _nome_valido(nome: str) -> bool:
    palavras = re.findall(r"[A-Za-zÀ-ÿ]{2,}", nome)
    return 2 <= len(palavras) <= 12 and len(nome) >= 5


def extrair_metadados(resultado: OCRResult) -> list[MetadataDetection]:
    """Preserva tokens e deriva campos úteis sem esconder o texto original."""
    motor = resultado.motor or "desconhecido"
    encontrados: list[MetadataDetection] = []

    def adicionar(
        tipo: str,
        original: str,
        confianca: float,
        *,
        tratado: str | None = None,
        bbox=None,
        contexto: str = "",
    ) -> None:
        valor = tratar_valor(tratado if tratado is not None else original)
        normalizado = normalizar_busca(valor)
        if not normalizado:
            return
        encontrados.append(
            MetadataDetection(
                tipo=tipo,
                valor_original=tratar_valor(original),
                valor_tratado=valor,
                valor_normalizado=normalizado,
                confianca=max(0.0, min(1.0, float(confianca))),
                motor=motor,
                bbox_json=json.dumps(bbox, ensure_ascii=False) if bbox else None,
                contexto=tratar_valor(contexto)[:300],
            )
        )

    # Cada token fornecido pelo motor é mantido, inclusive caixa e confiança.
    for token in resultado.tokens:
        adicionar(
            token.tipo or "token",
            token.valor,
            token.confianca,
            bbox=token.bbox,
        )

    texto = tratar_valor(resultado.texto_bruto)
    for linha in texto.splitlines():
        linha = tratar_valor(linha)
        if len(normalizar_busca(linha)) >= 2:
            adicionar("texto_linha", linha, 0.45)

    padroes_nome = (
        ("nome_registrado", r"(?i)(?:recebeu\s+o\s+nome\s+de|nome\s+de)\s*[:\-]?\s*([^\n,;]{5,110})", 0.72),
        ("declarante", r"(?i)em\s+cart[oó]rio\s+compareceu\s*[:\-]?\s*([^\n,;]{5,110})", 0.64),
        ("pai_ou_mae", r"(?i)filh[oa]\s+(?:leg[ií]tim[oa]\s+)?de\s*([^\n,;]{5,110})", 0.55),
    )
    for tipo, padrao, confianca in padroes_nome:
        for match in re.finditer(padrao, texto):
            nome = _limpar_nome(match.group(1))
            if _nome_valido(nome):
                adicionar(
                    tipo,
                    match.group(1),
                    confianca,
                    tratado=nome,
                    contexto=match.group(0),
                )

    # OCR de manuscrito costuma devolver o rótulo e o nome em linhas
    # separadas. Mantemos também essa leitura fraca para que ela possa ser
    # encontrada e corrigida pelo operador, sem tratá-la como confirmada.
    linhas = [tratar_valor(l) for l in texto.splitlines() if tratar_valor(l)]
    for indice, linha in enumerate(linhas):
        if not re.search(r"(?i)\bnome\s+de\b", linha):
            continue
        cauda = re.split(r"(?i)\bnome\s+de\b", linha, maxsplit=1)[-1]
        partes = [cauda] if cauda else []
        for proxima in linhas[indice + 1:indice + 5]:
            if _PARADAS_NOME.search(proxima):
                break
            limpa = re.sub(r"[^A-Za-zÀ-ÿ'´`\- ]", " ", proxima)
            limpa = re.sub(r"\s+", " ", limpa).strip()
            if len(limpa) >= 2:
                partes.append(limpa)
        nome = _limpar_nome(" ".join(partes))
        if _nome_valido(nome):
            adicionar(
                "nome_registrado",
                " | ".join(partes),
                0.48,
                tratado=nome,
                contexto="\n".join(linhas[indice:indice + 5]),
            )

    for match in re.finditer(r"(?<!\d)(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})(?!\d)", texto):
        adicionar("data", match.group(1), 0.75, contexto=match.group(0))
    for match in re.finditer(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)", texto):
        adicionar("ano", match.group(1), 0.65, contexto=match.group(0))

    # Remove repetições do mesmo motor, mas não elimina versões históricas de
    # execuções diferentes (isso é responsabilidade do banco).
    unicos: dict[tuple[str, str, str | None], MetadataDetection] = {}
    for item in encontrados:
        chave = (item.tipo, item.valor_normalizado, item.bbox_json)
        atual = unicos.get(chave)
        if atual is None or item.confianca > atual.confianca:
            unicos[chave] = item
    return list(unicos.values())
