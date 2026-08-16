from __future__ import annotations

import logging
from pathlib import Path

import cv2

from ..database.repository import Repository
from ..imaging.record_regions import recortar_registro
from ..metadata.extractor import extrair_metadados
from ..ocr.combiner import OCRCombiner
from ..ocr.engines import RapidOCRProvider, TesseractProvider

logger = logging.getLogger(__name__)


class MetadataIndexer:
    """Executa uma única tentativa de cada motor por assento.

    Tanto sucesso quanto falha ficam persistidos. Assim a fila contém somente
    trabalho realmente novo; uma falha vira revisão e não um laço automático.
    """

    def __init__(
        self,
        repo: Repository,
        settings=None,
        usar_htr: bool = False,
        usar_got: bool = False,
    ) -> None:
        self.repo = repo
        self.settings = settings
        self.usar_htr = usar_htr
        self.usar_got = usar_got
        self._got_provider = None
        self.combiner = OCRCombiner()
        self._init_providers()

    def _init_providers(self) -> None:
        tesseract = TesseractProvider(
            tesseract_path=(
                self.settings.get("ocr", "tesseract_path", "")
                if self.settings else None
            ),
            lang=self.settings.get("ocr", "lang", "por") if self.settings else "por",
        )
        rapid = RapidOCRProvider(apenas_cabecalhos=False)
        if tesseract.is_available():
            self.combiner.add_provider(tesseract)
        if rapid.is_available():
            self.combiner.add_provider(rapid)
        if self.usar_htr:
            try:
                from ..ocr.htr_engine import HTREngine

                htr = HTREngine()
                if htr.is_available():
                    self.combiner.add_provider(htr)
            except Exception as exc:
                logger.warning("HTR indisponível para indexação: %s", exc)
        if self.usar_got:
            try:
                from ..ocr.got_ocr_engine import GOTOCRProvider

                model_path = (
                    self.settings.get("ocr", "got_model_path", "")
                    if self.settings else ""
                )
                max_tokens = (
                    self.settings.get("ocr", "got_max_new_tokens", 384)
                    if self.settings else 384
                )
                got = GOTOCRProvider(
                    model_path=model_path or None,
                    permitir_download=True,
                    max_new_tokens=max_tokens,
                )
                if got.is_available():
                    self._got_provider = got
                    self.combiner.add_provider(got)
            except Exception as exc:
                logger.warning("GOT-OCR 2.0 indisponível para indexação: %s", exc)

    @property
    def motores(self) -> list[str]:
        return [provider.name for provider in self.combiner.providers]

    @staticmethod
    def _recortar_registro(image, indice: int, total: int):
        return recortar_registro(image, indice, total)

    def _registrar_falha(
        self,
        *,
        imagem_id: int,
        registro_id: int,
        motor: str,
        erro: str,
    ) -> None:
        self.repo.criar_execucao_ocr(
            imagem_id=imagem_id,
            registro_id=registro_id,
            motor=motor,
            texto_bruto="",
            sucesso=False,
            erro=erro,
        )
        if not self.repo.tem_revisao_pendente(imagem_id, "ocr_falha"):
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="ocr_falha",
                detalhes=(
                    f"OCR {motor} não produziu resultado. A fotografia foi preservada "
                    f"e não será processada novamente de forma automática. Erro: {erro}"
                ),
            )

    def indexar_imagem(self, imagem_id: int) -> dict:
        imagem_db = self.repo.get_imagem(imagem_id)
        if not imagem_db:
            return {"erro": "Imagem não encontrada", "imagem_id": imagem_id}
        if imagem_db.get("tipo_documento", "registro") != "registro":
            return {
                "imagem_id": imagem_id,
                "ignorado": True,
                "motivo": "Documento sem assentos",
            }

        registros = self.repo.sincronizar_registros_imagem(imagem_id)
        if not registros:
            return {"erro": "A imagem não possui assentos", "imagem_id": imagem_id}
        if not self.combiner.providers:
            for registro in registros:
                if not self.repo.get_execucao_ocr_ativa(
                    imagem_id=imagem_id,
                    registro_id=registro["id"],
                    motor="ocr-indisponivel",
                ):
                    self._registrar_falha(
                        imagem_id=imagem_id,
                        registro_id=registro["id"],
                        motor="ocr-indisponivel",
                        erro="Nenhum motor disponível",
                    )
            return {"erro": "Nenhum motor de OCR disponível", "imagem_id": imagem_id}

        pendentes_por_registro: dict[int, list] = {}
        reutilizadas = 0
        for registro in registros:
            pendentes = []
            for provider in self.combiner.providers:
                existente = self.repo.get_execucao_ocr_ativa(
                    imagem_id=imagem_id,
                    registro_id=registro["id"],
                    motor=provider.name,
                )
                if existente:
                    reutilizadas += 1
                else:
                    pendentes.append(provider)
            if pendentes:
                pendentes_por_registro[registro["id"]] = pendentes

        if not pendentes_por_registro:
            return {
                "imagem_id": imagem_id,
                "registros": len(registros),
                "deteccoes": 0,
                "nomes": [],
                "motores": self.motores,
                "reutilizadas": reutilizadas,
                "processada": False,
                "mensagem": "OCR já salvo; nenhuma execução repetida",
            }

        if self.usar_got and any(
            provider.name == "got-ocr2"
            for providers in pendentes_por_registro.values()
            for provider in providers
        ):
            if self._got_provider is None:
                return {
                    "erro": "GOT-OCR 2.0 indisponível neste ambiente",
                    "imagem_id": imagem_id,
                }
            try:
                self._got_provider.load()
            except Exception as exc:
                logger.exception("Falha ao preparar GOT-OCR 2.0")
                for registro in registros:
                    if any(
                        p.name == "got-ocr2"
                        for p in pendentes_por_registro.get(registro["id"], [])
                    ):
                        self._registrar_falha(
                            imagem_id=imagem_id,
                            registro_id=registro["id"],
                            motor="got-ocr2",
                            erro=str(exc),
                        )
                return {"erro": f"Falha no GOT-OCR 2.0: {exc}", "imagem_id": imagem_id}

        armazenamento = Path(imagem_db.get("caminho_armazenamento") or "")
        path = armazenamento if armazenamento.is_file() else Path(
            imagem_db.get("caminho_original") or ""
        )
        if not path.is_file():
            return {"erro": "Arquivo da imagem não encontrado", "imagem_id": imagem_id}
        image = cv2.imread(str(path))
        if image is None:
            return {"erro": "Não foi possível ler a imagem", "imagem_id": imagem_id}
        rotacao = int(imagem_db.get("rotacao_visualizacao") or 0)
        if rotacao == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotacao == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotacao == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        total_deteccoes = 0
        nomes: list[str] = []
        falhas = 0
        processadas = 0
        for registro in registros:
            providers = pendentes_por_registro.get(registro["id"], [])
            if not providers:
                continue
            recorte = self._recortar_registro(
                image, int(registro["indice_na_imagem"]), len(registros)
            )
            for provider in providers:
                try:
                    resultado = provider.recognize(recorte, fast=False)
                    texto = (resultado.texto_bruto or "").strip()
                    if not texto and not resultado.tokens:
                        raise RuntimeError("nenhum texto reconhecido")
                    execucao_id = self.repo.criar_execucao_ocr(
                        imagem_id=imagem_id,
                        registro_id=registro["id"],
                        motor=provider.name,
                        texto_bruto=texto,
                        tempo_ms=resultado.tempo_ms,
                        sucesso=True,
                    )
                    deteccoes = [d.to_dict() for d in extrair_metadados(resultado)]
                    self.repo.salvar_deteccoes_ocr(
                        execucao_id=execucao_id,
                        imagem_id=imagem_id,
                        registro_id=registro["id"],
                        deteccoes=deteccoes,
                    )
                    total_deteccoes += len(deteccoes)
                    nomes.extend(
                        d["valor_tratado"] for d in deteccoes
                        if d["tipo"] == "nome_registrado"
                    )
                    processadas += 1
                except Exception as exc:
                    logger.warning("OCR %s falhou na imagem %s: %s", provider.name, imagem_id, exc)
                    self._registrar_falha(
                        imagem_id=imagem_id,
                        registro_id=registro["id"],
                        motor=provider.name,
                        erro=str(exc),
                    )
                    falhas += 1

            if registro.get("termo") is not None:
                self.repo.salvar_metadado_tratado(
                    imagem_id=imagem_id,
                    registro_id=registro["id"],
                    tipo="termo",
                    valor=str(registro["termo"]),
                    confianca=0.98,
                    fonte="sequencia_livro",
                    status="inferido_sequencia",
                )

        return {
            "imagem_id": imagem_id,
            "registros": len(registros),
            "deteccoes": total_deteccoes,
            "nomes": nomes,
            "motores": self.motores,
            "execucoes_novas": processadas,
            "reutilizadas": reutilizadas,
            "falhas": falhas,
            "processada": processadas > 0 or falhas > 0,
        }
