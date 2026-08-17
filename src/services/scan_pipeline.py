from __future__ import annotations

import logging
import hashlib
from pathlib import Path
import unicodedata

import cv2
import numpy as np

from ..database.repository import Repository
from ..session.scan_session import ScanSession
from ..imaging.quality import avaliar_qualidade
from ..imaging.thumbnail import gerar_thumbnail
from ..imaging.preprocess import preprocess_for_ocr
from ..imaging.adaptive_layout import AdaptiveLayoutDetector
from ..imaging.storage_derivative import criar_derivada_armazenamento
from ..duplicate.hashing import compute_hashes, compute_sha256
from ..duplicate.detector import detectar_duplicidade
from ..ocr.engines import TesseractProvider, RapidOCRProvider
from ..ocr.htr_engine import HTREngine
from ..ocr.combiner import OCRCombiner
from ..ocr.name_candidates import NameCandidateIndexer
from ..metadata.extractor import extrair_metadados

logger = logging.getLogger(__name__)


class ScanPipeline:
    def __init__(self, repo: Repository, session: ScanSession, acervo_root: Path, settings=None) -> None:
        self.repo = repo
        self.session = session
        self.acervo_root = acervo_root
        self.settings = settings
        self.combiner = OCRCombiner()
        self._rapid_header = None
        self._duplicate_image_cache: dict[str, np.ndarray] = {}
        self._layout_detectors: dict[int, AdaptiveLayoutDetector] = {}
        self._init_ocr()

    def _layout_detector(self, livro_id: int) -> AdaptiveLayoutDetector:
        detector = self._layout_detectors.get(int(livro_id))
        if detector is None:
            path = self.acervo_root / "layout_templates" / f"livro_{int(livro_id)}.json"
            detector = AdaptiveLayoutDetector(path)
            self._layout_detectors[int(livro_id)] = detector
        return detector

    def _init_ocr(self) -> None:
        tess = TesseractProvider()
        rapid = RapidOCRProvider(apenas_cabecalhos=True)
        tess_enabled = self.settings.get("ocr", "tesseract_enabled", True) if self.settings else True
        rapid_enabled = self.settings.get("ocr", "rapidocr_enabled", True) if self.settings else True
        htr_enabled = self.settings.get("ocr", "htr_enabled", False) if self.settings else False
        if tess_enabled and tess.is_available():
            self.combiner.add_provider(tess)
            logger.info("Tesseract disponivel")
        if rapid_enabled and rapid.is_available():
            self.combiner.add_provider(rapid)
            self._rapid_header = rapid
            logger.info("RapidOCR disponivel")
        if htr_enabled:
            htr = HTREngine()
            if htr.is_available():
                self.combiner.add_provider(htr)
                logger.info("HTR disponivel no processamento automatico")

    @staticmethod
    def _normalizar_classificacao(texto: str) -> str:
        return "".join(
            char for char in unicodedata.normalize("NFD", texto or "")
            if unicodedata.category(char) != "Mn"
        ).lower()

    def _classificar_documento(self, image: np.ndarray) -> dict:
        """Separa face de registro de abertura/indice antes de atribuir termo.

        A decisao procura a estrutura impressa do formulario nas duas
        orientacoes. Numeros manuscritos isolados nunca bastam para classificar
        uma pagina como registro.
        """
        if self._rapid_header is None:
            return {
                "tipo": "registro",
                "rotacao": 0,
                "confianca": 0.0,
                "motivo": "classificador indisponivel; fluxo conservador legado",
                "texto": "",
            }
        melhor = None
        for rotacao in (0, 180):
            oriented = image if rotacao == 0 else cv2.rotate(image, cv2.ROTATE_180)
            result = self._rapid_header.recognize(oriented, fast=True)
            text = self._normalizar_classificacao(result.texto_bruto)
            numero_count = text.count("numero")
            structure_count = sum(
                marker in text
                for marker in ("em cartorio", "perante as", "averbac", "mil novecentos")
            )
            score = numero_count * 2 + structure_count
            candidate = {
                "tipo": "registro" if numero_count >= 2 or (numero_count >= 1 and structure_count >= 1) else "documento_nao_registro",
                "rotacao": rotacao,
                "confianca": min(1.0, score / 5.0),
                "motivo": (
                    f"formulario confirmado: {numero_count} cabecalho(s) Numero, "
                    f"{structure_count} marcador(es)"
                    if score >= 3
                    else "estrutura de dois registros nao encontrada"
                ),
                "texto": result.texto_bruto,
                "score": score,
            }
            if melhor is None or candidate["score"] > melhor["score"]:
                melhor = candidate
            if candidate["tipo"] == "registro":
                break
        return melhor

    def processar_imagem_imediato(self, image_path: str, on_status=None) -> dict:
        path = Path(image_path)
        if not path.exists():
            return {"erro": "Arquivo nao encontrado"}
        livro_id = self.session.livro_id
        if not livro_id:
            return {"erro": "Nenhum livro selecionado"}

        t0 = cv2.getTickCount()
        sha256 = compute_sha256(path)
        image = cv2.imread(str(path))
        if image is None:
            return {"erro": "Nao foi possivel ler a imagem"}

        armazenamento_path = None
        armazenamento_sha256 = None
        armazenamento_result = None
        armazenamento_erro = None
        try:
            armazenamento_result = criar_derivada_armazenamento(
                path,
                self._storage_path(livro_id, path),
                target_dpi=float(
                    self.settings.get("imaging", "storage_dpi", 300)
                    if self.settings else 300
                ),
                jpeg_quality=int(
                    self.settings.get("imaging", "storage_jpeg_quality", 80)
                    if self.settings else 80
                ),
            )
            armazenamento_path = armazenamento_result.output_path
            armazenamento_sha256 = armazenamento_result.output_sha256
        except Exception as exc:
            armazenamento_erro = str(exc)
            logger.exception("Falha ao criar derivada de armazenamento para %s", path)

        ordem = self.repo.get_total_imagens_livro(livro_id) + 1
        livro_atual = self.session.livro or self.repo.get_livro(livro_id) or {}
        registros_configurados = max(1, int(livro_atual.get("registros_por_face") or 1))
        layout = self._layout_detector(livro_id).classificar(
            image,
            page_number=ordem,
            expected_records=registros_configurados,
        )
        registros_para_face = (
            layout.records_per_face
            if layout.confidence >= AdaptiveLayoutDetector.STRUCTURE_MIN
            else registros_configurados
        )

        if on_status:
            on_status("classificacao", "Confirmando se e uma face de registro...")
        classificacao = self._classificar_documento(image)
        # O classificador OCR historico foi calibrado para o formulario de
        # dois assentos. O detector estrutural pode confirmar com seguranca
        # uma face de um assento e, nesse caso, deve prevalecer para evitar
        # que a pagina seja descartada como documento sem registro.
        if (
            classificacao["tipo"] != "registro"
            and layout.confidence >= AdaptiveLayoutDetector.STRUCTURE_MIN
            and layout.records_per_face in (1, 2)
        ):
            classificacao = {
                **classificacao,
                "tipo": "registro",
                "confianca": max(
                    float(classificacao.get("confianca") or 0.0),
                    float(layout.confidence),
                ),
                "motivo": (
                    f"layout estrutural confirmou {layout.records_per_face} "
                    f"assento(s); {classificacao.get('motivo', '')}"
                ).strip(),
            }
        tipo_documento = classificacao["tipo"]
        eh_registro = tipo_documento == "registro"

        if on_status:
            on_status("hash", "Calculando hashes...")
        phash, dhash = compute_hashes(image)

        if on_status:
            on_status("qualidade", "Analisando qualidade...")
        qualidade = avaliar_qualidade(
            image,
            exigir_margens="capturas_camera" in path.parts,
        )

        if on_status:
            on_status("thumb", "Gerando miniatura...")
        thumb_dir = self._thumb_dir(livro_id)
        thumb_name = f"{path.stem}_thumb.jpg"
        thumb_path = thumb_dir / thumb_name
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(thumb_path), gerar_thumbnail(image),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        total_imgs = [
            item for item in self.repo.get_imagens_livro(livro_id)
            if item.get("tipo_documento", "registro") == "registro"
        ]

        if on_status:
            on_status("duplicidade", "Verificando duplicidade...")
        dup = {"status": "unico", "confianca": 1.0, "duplicata_de": None}
        imagem_comparacao = None
        if total_imgs and eh_registro:
            def loader(p):
                cached = self._duplicate_image_cache.get(str(p))
                if cached is not None:
                    return cached
                carregada = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
                if carregada is not None:
                    self._guardar_imagem_comparacao(str(p), carregada)
                return carregada

            imagem_comparacao = cv2.resize(
                image, (512, 768), interpolation=cv2.INTER_AREA
            )
            dup = detectar_duplicidade(
                imagem_comparacao,
                total_imgs,
                loader,
                sha256_atual=sha256,
                phash_atual=phash,
                dhash_atual=dhash,
            )
        if imagem_comparacao is None:
            imagem_comparacao = cv2.resize(
                image, (512, 768), interpolation=cv2.INTER_AREA
            )
        self._guardar_imagem_comparacao(str(path), imagem_comparacao)

        face = (self.session.ultima_face or "frente") if eh_registro else "indeterminado"
        folha = self.session.ultima_folha if eh_registro else None
        termo_inicial, termo_final = (
            self.session.intervalo_termos_com(registros_para_face)
            if eh_registro and layout.confidence >= AdaptiveLayoutDetector.STRUCTURE_MIN
            else (self.session.intervalo_termos_atual if eh_registro else (None, None))
        )
        duplicata_ref = None
        if eh_registro and dup["status"] == "duplicata_confirmada" and dup.get("duplicata_de"):
            duplicata_ref = self.repo.get_imagem(dup["duplicata_de"])
            if duplicata_ref:
                face = duplicata_ref.get("face") or face
                folha = duplicata_ref.get("folha_estimada") or folha
                termo_inicial = duplicata_ref.get("termo_inicial")
                termo_final = duplicata_ref.get("termo_final")

        imagem_id = self.repo.registrar_imagem(
            livro_id=livro_id,
            ordem_captura=ordem,
            hash_perceptual=phash,
            dhash=dhash,
            sha256=sha256,
            caminho_original=str(path),
            caminho_armazenamento=armazenamento_path,
            sha256_armazenamento=armazenamento_sha256,
            caminho_thumb=str(thumb_path),
            tipo_documento=tipo_documento,
            rotacao_visualizacao=classificacao["rotacao"],
            folha_estimada=folha,
            face=face,
            qualidade_foco=qualidade["foco_valor"],
            qualidade_exposicao=qualidade["exposicao_valor"],
            qualidade_enquadramento=qualidade["enquadramento_status"],
            qualidade_orientacao=0,
            qualidade_status=qualidade["status_geral"],
            qualidade_oclusao=qualidade["oclusao_valor"],
            qualidade_motivos="; ".join(qualidade["motivos_refazer"]),
            duplicidade_status=dup["status"],
            duplicidade_confianca=dup["confianca"],
            duplicidade_ref=dup["duplicata_de"],
            termo_inicial=termo_inicial,
            termo_final=termo_final,
            registros_detectados=registros_para_face if eh_registro else None,
            layout_id=layout.layout_id,
            layout_confidence=layout.confidence,
            layout_method=layout.method,
            layout_reason=layout.reason,
            termo_final_decidido=termo_final,
            termo_status="inferido_sequencia" if termo_inicial is not None else "pendente",
            status="processando",
            precisa_revisao=1 if (
                (not eh_registro)
                or qualidade["repetir_captura"]
                or dup["status"] != "unico"
                or layout.needs_review
                or armazenamento_erro is not None
            ) else 0,
        )
        self.repo.sincronizar_registros_imagem(imagem_id)

        if not eh_registro:
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="classificar_documento",
                detalhes=(
                    "A foto nao tem a estrutura de uma face com dois registros. "
                    "Foi preservada sem folha/termo e a contagem nao avancou. "
                    f"Motivo: {classificacao['motivo']}"
                ),
            )

        if armazenamento_erro:
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="armazenamento",
                detalhes=(
                    "Nao foi possivel criar a copia de armazenamento sem alterar "
                    f"o original: {armazenamento_erro}"
                ),
            )

        if dup["status"] == "duplicata_confirmada":
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="duplicidade",
                detalhes=f"Duplicata confirmada da imagem {dup['duplicata_de']} (SSIM: {dup.get('ssim', 0):.2f})",
            )
        if qualidade["repetir_captura"]:
            faixa = (
                str(termo_inicial)
                if termo_inicial == termo_final
                else f"{termo_inicial}-{termo_final}"
            )
            motivos = ", ".join(qualidade["motivos_refazer"])
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="refazer_captura",
                detalhes=(
                    f"Folha {folha or '?'} - {face.capitalize()} - termos {faixa} | "
                    f"Refotografar: {motivos}"
                ),
            )

        if (
            eh_registro
            and layout.needs_review
            and layout.confidence >= AdaptiveLayoutDetector.STRUCTURE_MIN
            and not self.repo.tem_revisao_pendente(imagem_id, "layout_incerto")
        ):
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="layout_incerto",
                detalhes=(
                    f"{layout.reason}; layout={layout.layout_id or 'desconhecido'}, "
                    f"registros detectados={layout.records_per_face}, "
                    f"confianca={layout.confidence:.2f}. Validar os crops antes de usar o lote."
                ),
            )

        # Uma duplicata nao representa uma nova face. Em caso apenas suspeito,
        # a sessao fica parada ate o operador confirmar ou rejeitar no app.
        sessao_avancou = eh_registro and dup["status"] == "unico"
        if sessao_avancou:
            self.session.avancar_pagina(registros_na_face=registros_para_face)
        elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000

        from .telemetry import emitir
        emitir("capture.saved", imagem_id=imagem_id, duration_ms=round(elapsed, 1),
               qualidade="repetir" if qualidade["repetir_captura"] else "ok",
               duplicidade=dup["status"], bytes=int(armazenamento_result.output_size_bytes or 0)
               if armazenamento_result else 0)

        return {
            "imagem_id": imagem_id,
            "phash": phash,
            "sha256": sha256[:16],
            "qualidade": qualidade,
            "duplicidade": dup,
            "folha": folha,
            "face": face,
            "termo_inicial": termo_inicial,
            "termo_final": termo_final,
            "proximo_termo": self.session.proximo_termo_esperado,
            # Mantido por compatibilidade com consumidores antigos. Agora aponta
            # para o primeiro termo desta imagem, nao para a proxima imagem.
            "termo_esperado": termo_inicial,
            "tempo_ms": round(elapsed, 1),
            "thumb": str(thumb_path),
            "caminho_armazenamento": armazenamento_path,
            "sha256_armazenamento": armazenamento_sha256,
            "armazenamento": armazenamento_result.to_dict() if armazenamento_result else None,
            "armazenamento_erro": armazenamento_erro,
            "precisa_revisao": (
                qualidade["repetir_captura"]
                or dup["status"] != "unico"
                or layout.needs_review
            ),
            "aguarda_confirmacao_duplicidade": dup["status"] == "possivel_duplicata",
            "sessao_avancou": sessao_avancou,
            "tipo_documento": tipo_documento,
            "nao_registro": not eh_registro,
            "classificacao": classificacao,
            "rotacao_visualizacao": classificacao["rotacao"],
            "layout": layout.to_dict(),
            "registros_detectados": registros_para_face if eh_registro else 0,
        }

    def resolver_possivel_duplicata(self, imagem_id: int, confirmar: bool) -> dict:
        """Resolve a pergunta de duplicidade antes de avancar a sessao."""
        imagem = self.repo.get_imagem(imagem_id)
        if not imagem or imagem.get("duplicidade_status") != "possivel_duplicata":
            return imagem or {"erro": "Imagem nao encontrada"}

        if confirmar:
            ref = self.repo.get_imagem(imagem.get("duplicidade_ref"))
            updates = {"duplicidade_status": "duplicata_confirmada", "precisa_revisao": 1}
            if ref:
                updates.update(
                    termo_inicial=ref.get("termo_inicial"),
                    termo_final=ref.get("termo_final"),
                    termo_final_decidido=ref.get("termo_final_decidido") or ref.get("termo_final"),
                    folha_estimada=ref.get("folha_estimada"),
                    face=ref.get("face"),
                )
            self.repo.atualizar_imagem(imagem_id, **updates)
            self.repo.criar_revisao(
                imagem_id=imagem_id,
                tipo="duplicidade",
                detalhes=f"Duplicata confirmada pelo operador da imagem {imagem.get('duplicidade_ref')}",
            )
        else:
            # A qualidade continua pendente mesmo quando a suspeita de
            # duplicidade e descartada.
            pendencia_qualidade = self.repo.tem_revisao_pendente(
                imagem_id, "refazer_captura"
            )
            precisa_revisao = 1 if pendencia_qualidade else 0
            self.repo.atualizar_imagem(
                imagem_id,
                duplicidade_status="unico",
                duplicidade_confianca=0.0,
                duplicidade_ref=None,
                precisa_revisao=precisa_revisao,
            )
            self.session.avancar_pagina(
                registros_na_face=imagem.get("registros_detectados")
            )

        return self.repo.get_imagem(imagem_id)

    def substituir_captura(
        self,
        imagem_id: int,
        revisao_id: int,
        novo_path: str,
    ) -> dict:
        """Troca somente a fotografia de uma face ja contada.

        Folha, face e termos permanecem os mesmos. Assim uma refotografia feita
        no fim do lote nunca avanca a sessao nem cria um termo inexistente.
        """
        registrada = self.repo.get_imagem(imagem_id)
        path = Path(novo_path)
        if not registrada:
            return {"erro": "Registro original nao encontrado"}
        if not path.is_file():
            return {"erro": "Nova fotografia nao encontrada"}
        image = cv2.imread(str(path))
        if image is None:
            return {"erro": "Nao foi possivel ler a nova fotografia"}

        qualidade = avaliar_qualidade(
            image,
            exigir_margens=(
                "capturas_camera" in path.parts or "refotos_camera" in path.parts
            ),
        )
        thumb_dir = self._thumb_dir(registrada["livro_id"])
        thumb_path = thumb_dir / f"refoto_{imagem_id}_{path.stem}_thumb.jpg"
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(thumb_path), gerar_thumbnail(image),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        phash, dhash = compute_hashes(image)
        updates = {
            "hash_perceptual": phash,
            "dhash": dhash,
            "sha256": compute_sha256(path),
            "caminho_original": str(path),
            "caminho_armazenamento": None,
            "sha256_armazenamento": None,
            "caminho_thumb": str(thumb_path),
            "qualidade_foco": qualidade["foco_valor"],
            "qualidade_exposicao": qualidade["exposicao_valor"],
            "qualidade_enquadramento": qualidade["enquadramento_status"],
            "qualidade_status": qualidade["status_geral"],
            "qualidade_oclusao": qualidade["oclusao_valor"],
            "qualidade_motivos": "; ".join(qualidade["motivos_refazer"]),
            "status": "pendente_ocr",
        }
        try:
            armazenamento = criar_derivada_armazenamento(
                path,
                self._storage_path(int(registrada["livro_id"]), path),
                target_dpi=float(
                    self.settings.get("imaging", "storage_dpi", 300)
                    if self.settings else 300
                ),
                jpeg_quality=int(
                    self.settings.get("imaging", "storage_jpeg_quality", 80)
                    if self.settings else 80
                ),
            )
            updates["caminho_armazenamento"] = armazenamento.output_path
            updates["sha256_armazenamento"] = armazenamento.output_sha256
        except Exception:
            logger.exception("Falha ao atualizar derivada de armazenamento da imagem %s", imagem_id)
        self.repo.invalidar_ocr_imagem(imagem_id)
        self.repo.atualizar_imagem(imagem_id, **updates)

        folha = registrada.get("folha_estimada") or "?"
        face = (registrada.get("face") or "indeterminada").capitalize()
        termo_i, termo_f = registrada.get("termo_inicial"), registrada.get("termo_final")
        faixa = "?" if termo_i is None else (
            str(termo_i) if termo_i == termo_f else f"{termo_i}-{termo_f}"
        )
        if qualidade["repetir_captura"]:
            motivos = ", ".join(qualidade["motivos_refazer"])
            self.repo.atualizar_imagem(imagem_id, precisa_revisao=1)
            self.repo.atualizar_revisao(
                revisao_id,
                f"Folha {folha} - {face} - termos {faixa} | Refotografar: {motivos}",
            )
        else:
            self.repo.resolver_revisao(revisao_id)

        return {
            "imagem_id": imagem_id,
            "qualidade": qualidade,
            "thumb": str(thumb_path),
            "substituida": not qualidade["repetir_captura"],
            "folha": registrada.get("folha_estimada"),
            "face": registrada.get("face"),
            "termo_inicial": termo_i,
            "termo_final": termo_f,
            "ocr_pendente": True,
        }

    def processar_ocr_secundario(self, imagem_id: int, on_status=None) -> dict:
        img_data = self.repo.get_imagem(imagem_id)
        if not img_data:
            return {"erro": "Imagem nao encontrada"}

        # Este marcador representa o ato completo de OCR da fotografia. Se ele
        # existe, texto, campos, tempos e alertas já estão no banco; abrir a
        # imagem novamente ou recolocá-la na fila não executa OCR outra vez.
        marcador = self.repo.get_execucao_ocr_ativa(
            imagem_id=imagem_id,
            registro_id=None,
            motor="pipeline-ocr-v1",
        )
        if marcador:
            execucoes = self.repo.listar_execucoes_ocr_ativas(
                imagem_id=imagem_id,
                registro_id=None,
            )
            return {
                "imagem_id": imagem_id,
                "termo": img_data.get("termo_final_decidido"),
                "termo_inicial": img_data.get("termo_inicial"),
                "termo_final": img_data.get("termo_final"),
                "termo_status": img_data.get("termo_status") or "concluido",
                "termo_confianca": round(float(img_data.get("confianca_termo") or 0), 3),
                "folha": img_data.get("folha_final_decidida"),
                "folha_status": img_data.get("folha_status") or "nao_identificado",
                "motor": img_data.get("motor_utilizado") or "salvo",
                "caminho_processamento": (
                    img_data.get("caminho_armazenamento")
                    or img_data.get("caminho_original")
                ),
                "tempos_ms": {
                    e["motor"]: round(float(e.get("tempo_ms") or 0), 1)
                    for e in execucoes
                    if e["motor"] != "pipeline-ocr-v1"
                },
                "reutilizado": True,
                "mensagem": "OCR já processado; resultado carregado do banco",
            }

        armazenamento = Path(img_data.get("caminho_armazenamento") or "")
        path = armazenamento if armazenamento.is_file() else Path(
            img_data["caminho_original"]
        )
        if not path.exists():
            return {"erro": "Arquivo de imagem nao encontrado"}

        image = cv2.imread(str(path))
        if image is None:
            return {"erro": "Nao foi possivel ler a imagem"}

        if img_data.get("tipo_documento", "registro") != "registro":
            return {
                "imagem_id": imagem_id,
                "ignorado": True,
                "motivo": "Documento preservado sem OCR de termo",
                "tipo_documento": img_data.get("tipo_documento"),
            }

        rotacao = int(img_data.get("rotacao_visualizacao") or 0)
        if rotacao == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotacao == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotacao == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if on_status:
            on_status("ocr", "Executando OCR...")
        image_ocr = preprocess_for_ocr(image, str(path))

        # O worker pode terminar depois de outras imagens terem sido capturadas.
        # Por isso a faixa esperada deve vir do registro imutavel da imagem, e
        # nunca do estado atual (ja avancado) da sessao.
        esperado_min = img_data.get("termo_inicial")
        esperado_max = img_data.get("termo_final")

        results = self.combiner.recognize_all(image_ocr, fast=False)
        self._persistir_resultados_ocr(imagem_id, results)
        registros_nome = self.repo.sincronizar_registros_imagem(imagem_id)
        provedores_nome = [
            provider for provider in self.combiner.providers
            if provider.name in {"tesseract", "rapidocr"}
        ]
        nomes_rapidos = (
            NameCandidateIndexer(
                self.repo,
                provedores_nome,
                limiar_qwen=float(
                    self.settings.get("ocr", "name_qwen_threshold", 0.78)
                    if self.settings else 0.78
                ),
            ).indexar(imagem_id, image, registros_nome)
            if provedores_nome else {"nomes": [], "incertos": [], "reutilizados": 0}
        )
        termo_result = self.combiner.extrair_termo(
            results,
            esperado_min,
            esperado_max,
            fallback_sequencia=True,
        )
        folha_result = self.combiner.extrair_folha(results)

        motor = termo_result.motor_principal or "desconhecido"
        ocr_valor = termo_result.valor
        if motor == "sequencia":
            ocr_valor = (
                termo_result.alternativas[0].get("valor")
                if termo_result.alternativas else None
            )

        folha_status = "nao_identificado"
        if folha_result.valor is not None:
            folha_status = "identificado" if folha_result.confianca >= 0.7 else "inferido"

        updates = {
            "ocr_termo": str(ocr_valor) if ocr_valor is not None else "",
            "ocr_folha": str(folha_result.valor) if folha_result.valor else "",
            # A faixa sequencial continua sendo a decisao principal. O OCR e
            # uma validacao e fica preservado separadamente em ocr_termo.
            "termo_final_decidido": esperado_max,
            "folha_final_decidida": folha_result.valor,
            "termo_status": termo_result.status,
            "folha_status": folha_status,
            "motor_utilizado": motor,
            "confianca_termo": termo_result.confianca,
            "confianca_folha": folha_result.confianca,
            "status": "concluido",
            "precisa_revisao": 1 if termo_result.status == "precisa_revisao" else img_data.get("precisa_revisao", 0),
        }
        self.repo.atualizar_imagem(imagem_id, **updates)
        registros = self.repo.sincronizar_registros_imagem(imagem_id)
        for registro in registros:
            if registro.get("termo") is not None:
                self.repo.salvar_metadado_tratado(
                    imagem_id=imagem_id,
                    registro_id=registro["id"],
                    tipo="termo",
                    valor=str(registro["termo"]),
                    confianca=0.98,
                    fonte="sequencia_livro",
                    motor=motor,
                    status="inferido_sequencia",
                )
            folha_valor = folha_result.valor or registro.get("folha")
            if folha_valor is not None:
                self.repo.salvar_metadado_tratado(
                    imagem_id=imagem_id,
                    registro_id=registro["id"],
                    tipo="folha",
                    valor=str(folha_valor),
                    confianca=folha_result.confianca if folha_result.valor else 0.90,
                    fonte="ocr" if folha_result.valor else "sequencia_livro",
                    motor=folha_result.motor_principal,
                    status=folha_status,
                )

        if termo_result.status in ("duvidoso", "precisa_revisao"):
            lido = ocr_valor if ocr_valor is not None else "nao identificado"
            if not self.repo.tem_revisao_pendente(imagem_id, "termo_incerto"):
                self.repo.criar_revisao(
                    imagem_id=imagem_id,
                    tipo="termo_incerto",
                    detalhes=f"Termo OCR: {lido} | Esperado: {esperado_min}-{esperado_max} | Status: {termo_result.status} | Confianca: {termo_result.confianca:.2f} | Texto: {termo_result.texto_bruto[:200]}",
                )
        if folha_result.status == "duvidoso":
            if not self.repo.tem_revisao_pendente(imagem_id, "folha_incerta"):
                self.repo.criar_revisao(
                    imagem_id=imagem_id,
                    tipo="folha_incerta",
                    detalhes=f"Folha encontrada: {folha_result.valor} | Confianca: {folha_result.confianca:.2f}",
                )

        tempos = {r.motor: round(r.tempo_ms, 1) for r in results if r.tempo_ms > 0}

        # Gravado por último: sua presença garante que todo o ato anterior foi
        # concluído. Uma refotografia invalida este marcador e permite uma única
        # execução para a nova imagem.
        self.repo.criar_execucao_ocr(
            imagem_id=imagem_id,
            registro_id=None,
            motor="pipeline-ocr-v1",
            texto_bruto="",
            tempo_ms=sum(tempos.values()),
            sucesso=True,
        )

        from .telemetry import emitir
        emitir("ocr.secondary_finished", imagem_id=imagem_id,
               duration_ms=round(sum(tempos.values()), 1),
               sucesso=True, motor=motor,
               uncertain=bool(nomes_rapidos.get("incertos")))

        return {
            "imagem_id": imagem_id,
            "termo": termo_result.valor,
            "termo_inicial": esperado_min,
            "termo_final": esperado_max,
            "termo_status": termo_result.status,
            "termo_confianca": round(termo_result.confianca, 3),
            "folha": folha_result.valor,
            "folha_status": folha_status,
            "motor": motor,
            "caminho_processamento": str(path),
            "tempos_ms": tempos,
            "reutilizado": False,
            "nomes_rapidos": nomes_rapidos.get("nomes", []),
            "nomes_incerto": nomes_rapidos.get("incertos", []),
        }

    def _persistir_resultados_ocr(self, imagem_id: int, results: list) -> None:
        """Guarda texto, tokens e campos derivados com sua procedência."""
        registros = self.repo.sincronizar_registros_imagem(imagem_id)
        por_termo = {
            int(r["termo"]): r for r in registros if r.get("termo") is not None
        }
        for result in results:
            execucao_id = self.repo.criar_execucao_ocr(
                imagem_id=imagem_id,
                registro_id=None,
                motor=result.motor or "desconhecido",
                texto_bruto=result.texto_bruto or "",
                tempo_ms=result.tempo_ms,
                sucesso=True,
            )
            gerais = []
            especificas: dict[int, list[dict]] = {}
            for deteccao in extrair_metadados(result):
                item = deteccao.to_dict()
                registro = None
                if deteccao.tipo == "termo" and deteccao.valor_normalizado.isdigit():
                    registro = por_termo.get(int(deteccao.valor_normalizado))
                if registro:
                    especificas.setdefault(registro["id"], []).append(item)
                else:
                    gerais.append(item)
            self.repo.salvar_deteccoes_ocr(
                execucao_id=execucao_id,
                imagem_id=imagem_id,
                registro_id=None,
                deteccoes=gerais,
            )
            for registro_id, deteccoes in especificas.items():
                self.repo.salvar_deteccoes_ocr(
                    execucao_id=execucao_id,
                    imagem_id=imagem_id,
                    registro_id=registro_id,
                    deteccoes=deteccoes,
                )

    def _thumb_dir(self, livro_id: int) -> Path:
        d = self.acervo_root / f"livro_{livro_id}" / "thumbs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _storage_path(self, livro_id: int, source_path: Path) -> Path:
        target_dpi = int(
            self.settings.get("imaging", "storage_dpi", 300)
            if self.settings else 300
        )
        return (
            self.acervo_root
            / f"livro_{livro_id}"
            / f"armazenamento_{target_dpi}dpi"
            / f"{source_path.stem}.jpg"
        )

    def _guardar_imagem_comparacao(self, path: str, image: np.ndarray) -> None:
        self._duplicate_image_cache[path] = image
        while len(self._duplicate_image_cache) > 4:
            primeira = next(iter(self._duplicate_image_cache))
            del self._duplicate_image_cache[primeira]
