from __future__ import annotations

import logging
import os
from pathlib import Path
import csv
import json
import re
import sys

import cv2
import numpy as np

from PySide6.QtCore import Qt, QProcess, QThread, Signal, Slot, QUrl, QTimer
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config.settings import APP_VERSION, Settings, data_dir
from ..database.connection import Database
from ..database.repository import Repository
from ..imaging.document import retificar_formulario
from ..imaging.oriented_copy import aplicar_rotacao, materializar_copia_orientada, normalizar_rotacao
from ..imaging.record_regions import (
    bbox_contido_no_registro,
    bbox_corresponde_registro,
    bbox_registro,
)
from ..metadata.normalizer import normalizar_busca, tratar_valor
from ..ocr.combiner import texto_evidencia_termo
from ..ocr.qwen_vl_engine import (
    MODEL_SIZE_MIB as QWEN_MODEL_SIZE_MIB,
    QwenAreaAnalyzer,
    QwenRecordAnalyzer,
    modelo_qwen_instalado,
    preparar_imagem_qwen,
)
from ..ocr.name_candidates import recortar_registro_nome
from ..services.metadata_indexer import MetadataIndexer
from ..services.job_context import QwenJobContext, validar_contexto_qwen
from ..services.acervo_api import AcervoApiServer
from ..services.organized_book_importer import OrganizedBookImporter, auditar_a07
from ..services.generic_book_importer import (
    BookAudit,
    BookImportSpec,
    GenericBookImporter,
    auditar_livro,
)
from ..ui.image_reader_window import ImageReaderWindow
from ..ui.processing_dialog import ProcessingDialog

logger = logging.getLogger(__name__)


TIPOS_METADADO = {
    "nome_registrado": "Nome do registrado",
    "nome_mae": "Nome da mãe",
    "declarante": "Declarante",
    "pai_ou_mae": "Filiação",
    "data": "Data",
    "ano": "Ano",
    "termo": "Termo",
    "folha": "Folha",
    "local": "Local",
    "texto_linha": "Texto OCR",
}

PROMPTS_QWEN = {
    "Nome do registrado": (
        "nome_registrado",
        "A imagem contém somente um nome manuscrito de registro civil em português. "
        "Transcreva exatamente o nome completo, mantendo os espaços. Responda somente "
        "com o nome, sem explicação e sem tentar completar partes ilegíveis.",
    ),
    "Número do termo": (
        "termo",
        "Leia somente o número manuscrito do termo nesta pequena área. Responda apenas "
        "com os algarismos, sem explicação.",
    ),
    "Data": (
        "data",
        "Transcreva exatamente a data manuscrita nesta pequena área de registro civil. "
        "Responda somente com a data, sem explicação.",
    ),
    "Filiação": (
        "pai_ou_mae",
        "Transcreva exatamente o nome manuscrito de pai ou mãe nesta pequena área. "
        "Responda somente com o texto lido, sem explicação.",
    ),
    "Outro texto": (
        "texto_linha",
        "Transcreva exatamente apenas o texto manuscrito visível nesta pequena área. "
        "Não explique, não resuma e não complete o que estiver ilegível.",
    ),
}


def _mesma_area(
    primeira: tuple[float, float, float, float] | list[float],
    segunda: tuple[float, float, float, float] | list[float],
    tolerancia: float = 0.025,
) -> bool:
    return len(primeira) == 4 and len(segunda) == 4 and all(
        abs(float(a) - float(b)) <= tolerancia
        for a, b in zip(primeira, segunda)
    )


def _distancia_texto(primeiro: str, segundo: str) -> int:
    linhas = list(range(len(segundo) + 1))
    for i, caractere in enumerate(primeiro, 1):
        atual = [i]
        for j, outro in enumerate(segundo, 1):
            atual.append(min(
                atual[-1] + 1,
                linhas[j] + 1,
                linhas[j - 1] + (caractere != outro),
            ))
        linhas = atual
    return linhas[-1]


def _tratar_termo_qwen(valor: str, esperado: int | None) -> tuple[str, str]:
    """Retira pontuação e usa a sequência auditada apenas para corrigir ruído."""
    bruto = str(valor or "").strip()
    digitos = re.sub(r"\D", "", bruto)
    if esperado is None or not digitos:
        return (digitos or bruto), ""
    esperado_texto = str(int(esperado))
    if digitos == esperado_texto:
        return esperado_texto, ""
    if _distancia_texto(digitos, esperado_texto) <= 1:
        return esperado_texto, f"Qwen leu {bruto!r}; ajustado pela sequência auditada"
    return digitos, f"Qwen leu {bruto!r}; confirmar contra a sequência"


def estado_visual_registro(registro: dict, tem_revisao: bool = False) -> dict:
    """Uma sugestão nunca pode ser apresentada como confirmação."""
    if tem_revisao:
        return {
            "texto": "● REVISAR — há pendência nesta foto",
            "fundo": "#ffebee",
            "frente": "#b71c1c",
        }
    if registro.get("nome_confirmado"):
        return {
            "texto": "● CONFIRMADO — nome conferido por operador",
            "fundo": "#e8f5e9",
            "frente": "#1b5e20",
        }
    if registro.get("nome_sugerido"):
        return {
            "texto": (
                "SUGESTÃO NÃO CONFIRMADA — "
                f"{float(registro.get('nome_confianca') or 0) * 100:.0f}% · "
                f"{registro.get('nome_fonte') or 'OCR'}"
            ),
            "fundo": "#ffffff",
            "frente": "#37474f",
        }
    return {
        "texto": "● INFORMAÇÃO — abra o revisor para conferir",
        "fundo": "#e3f2fd",
        "frente": "#0d47a1",
    }


class IndexacaoWorker(QThread):
    progresso = Signal(int, int, str)
    imagem_concluida = Signal(dict)
    erro = Signal(str)
    concluido = Signal(int, int)

    def __init__(
        self,
        repo: Repository,
        settings: Settings,
        imagens: list[dict],
    ) -> None:
        super().__init__()
        self.repo = repo
        self.settings = settings
        self.imagens = imagens

    def run(self) -> None:
        try:
            indexador = MetadataIndexer(
                self.repo,
                self.settings,
            )
            if not indexador.motores:
                self.erro.emit("Nenhum motor de OCR está disponível.")
                return
            processadas = 0
            falhas = 0
            total = len(self.imagens)
            for numero, imagem in enumerate(self.imagens, 1):
                if self.isInterruptionRequested():
                    break
                nome = Path(imagem.get("caminho_original") or "").name
                self.progresso.emit(numero, total, nome)
                resultado = indexador.indexar_imagem(imagem["id"])
                if resultado.get("erro"):
                    falhas += 1
                else:
                    processadas += 1
                self.imagem_concluida.emit(resultado)
            self.concluido.emit(processadas, falhas)
        except Exception as exc:
            logger.exception("Falha na indexação")
            self.erro.emit(str(exc))


class QwenAreaWorker(QThread):
    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(
        self,
        imagem,
        tipo: str,
        instrucao: str,
        bbox_relativo: tuple[float, float, float, float],
        model_path: str | None,
        max_new_tokens: int,
        job_context: QwenJobContext,
        preprocessar: bool = False,
    ) -> None:
        super().__init__()
        self.imagem = imagem.copy()
        self.tipo = tipo
        self.instrucao = instrucao
        self.bbox_relativo = bbox_relativo
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.job_context = job_context
        self.preprocessar = bool(preprocessar)

    def run(self) -> None:
        analisador = None
        try:
            analisador = QwenAreaAnalyzer(
                model_path=self.model_path or None,
                permitir_download=True,
                max_new_tokens=self.max_new_tokens,
            )
            imagem_modelo = preparar_imagem_qwen(self.imagem) if self.preprocessar else self.imagem
            resultado = analisador.analisar(
                imagem_modelo,
                instrucao=self.instrucao,
                tipo=self.tipo,
            )
            if not resultado.texto_bruto.strip():
                raise RuntimeError("O Qwen não conseguiu ler texto nessa área.")
            self.concluido.emit({
                "tipo": self.tipo,
                "texto": resultado.texto_bruto,
                "motor": resultado.motor,
                "tempo_ms": resultado.tempo_ms,
                "bbox_relativo": self.bbox_relativo,
                "job_context": self.job_context.to_dict(),
            })
        except Exception as exc:
            logger.exception("Falha na leitura de área com Qwen")
            self.erro.emit(str(exc))
        finally:
            if analisador is not None:
                analisador.liberar()


class QwenRecordWorker(QThread):
    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(self, imagem, job_context: QwenJobContext, model_path, max_new_tokens) -> None:
        super().__init__()
        self.imagem = imagem.copy()
        self.job_context = job_context
        self.bbox_relativo = job_context.bbox
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

    def run(self) -> None:
        analisador = None
        try:
            imagem_modelo = retificar_formulario(self.imagem).image
            altura, largura = imagem_modelo.shape[:2]
            x1 = max(0, min(largura - 1, int(self.bbox_relativo[0] * largura)))
            y1 = max(0, min(altura - 1, int(self.bbox_relativo[1] * altura)))
            x2 = max(x1 + 1, min(largura, int(self.bbox_relativo[2] * largura)))
            y2 = max(y1 + 1, min(altura, int(self.bbox_relativo[3] * altura)))
            recorte = imagem_modelo[y1:y2, x1:x2]
            if recorte.shape[0] < 80 or recorte.shape[1] < 100:
                raise RuntimeError("A área automática do registro ficou pequena demais.")
            analisador = QwenRecordAnalyzer(
                model_path=self.model_path or None,
                permitir_download=True,
                max_new_tokens=self.max_new_tokens,
            )
            campos, resultado = analisador.analisar_registro(recorte)
            self.concluido.emit({
                "campos": campos,
                "texto_bruto": resultado.texto_bruto,
                "motor": resultado.motor,
                "tempo_ms": resultado.tempo_ms,
                "bbox_relativo": self.bbox_relativo,
                "job_context": self.job_context.to_dict(),
            })
        except Exception as exc:
            logger.exception("Falha na leitura estruturada do registro com Qwen")
            self.erro.emit(str(exc))
        finally:
            if analisador is not None:
                analisador.liberar()


class QwenNomeBatchWorker(QThread):
    """Corrige somente nomes que a primeira passada marcou como incertos."""

    item_concluido = Signal(dict)
    concluido = Signal(int, int)
    erro = Signal(str)

    def __init__(self, itens: list[dict], model_path: str | None, max_new_tokens: int) -> None:
        super().__init__()
        self.itens = list(itens)
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

    def run(self) -> None:
        analisador = None
        processados = 0
        falhas = 0
        try:
            analisador = QwenRecordAnalyzer(
                model_path=self.model_path or None,
                permitir_download=True,
                max_new_tokens=self.max_new_tokens,
            )
            for item in self.itens:
                if self.isInterruptionRequested():
                    break
                try:
                    caminho = Path(item.get("caminho_original") or "")
                    dados = np.fromfile(str(caminho), dtype=np.uint8)
                    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
                    if imagem is None:
                        raise RuntimeError("não foi possível abrir a fotografia")
                    registros_total = max(
                        1,
                        int(item.get("termo_final") or item.get("termo_inicial") or item.get("termo") or 1)
                        - int(item.get("termo_inicial") or item.get("termo") or 1)
                        + 1,
                    )
                    recorte = recortar_registro_nome(
                        imagem,
                        int(item.get("indice_na_imagem") or 0),
                        registros_total,
                    )
                    campos, resultado = analisador.analisar_registro(recorte)
                    self.item_concluido.emit({
                        "item": item,
                        "nome": str(campos.get("nome_registrado") or "").strip(),
                        "campos": campos,
                        "texto_bruto": resultado.texto_bruto,
                        "tempo_ms": resultado.tempo_ms,
                        "motor": resultado.motor,
                    })
                    processados += 1
                except Exception as exc:
                    falhas += 1
                    logger.exception("Qwen falhou no nome do registro %s", item.get("registro_id"))
            self.concluido.emit(processados, falhas)
        except Exception as exc:
            logger.exception("Falha ao iniciar fila de nomes Qwen")
            self.erro.emit(str(exc))
        finally:
            if analisador is not None:
                analisador.liberar()


class ImportacaoA07Worker(QThread):
    progresso = Signal(int, int, str)
    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(self, repo: Repository, root: Path) -> None:
        super().__init__()
        self.repo = repo
        self.root = root

    def run(self) -> None:
        try:
            importer = OrganizedBookImporter(self.repo)
            result = importer.importar_a07(
                self.root,
                on_progress=lambda current, total, label: self.progresso.emit(
                    current, total, label
                ),
            )
            self.concluido.emit(result)
        except Exception as exc:
            logger.exception("Falha ao importar o Livro A-07")
            self.erro.emit(str(exc))


class ImportacaoLivroWorker(QThread):
    progresso = Signal(int, int, str)
    auditoria_concluida = Signal(object)
    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(self, db_path: Path, root: Path, audit: BookAudit | None = None) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.root = Path(root)
        self.audit = audit

    def run(self) -> None:
        try:
            if self.audit is None:
                audit = auditar_livro(
                    self.root,
                    BookImportSpec.a16(),
                    on_progress=lambda current, total, label: self.progresso.emit(
                        current, total, label
                    ),
                )
                self.auditoria_concluida.emit(audit)
                return
            db = Database(self.db_path)
            try:
                db.connect()
                importer = GenericBookImporter(
                    Repository(db), normalized_root=data_dir() / "normalizadas"
                )
                result = importer.importar(
                    self.audit,
                    on_progress=lambda current, total, label: self.progresso.emit(
                        current, total, label
                    ),
                    should_stop=self.isInterruptionRequested,
                )
                self.concluido.emit(result)
            finally:
                db.close()
        except Exception as exc:
            logger.exception("Falha na auditoria/importação do livro organizado")
            self.erro.emit(str(exc))


class SincronizacaoWorker(QThread):
    """Materializa registros legados sem bloquear a abertura da Consulta."""

    concluido = Signal(int)
    erro = Signal(str)

    def run(self) -> None:
        db = Database()
        try:
            db.connect()
            total = Repository(db).sincronizar_todos_registros()
            self.concluido.emit(total)
        except Exception as exc:
            logger.exception("Falha ao sincronizar registros legados")
            self.erro.emit(str(exc))
        finally:
            db.close()


class ConsultaMainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.db = Database()
        self.db.connect()
        self.repo = Repository(self.db)
        self._associacoes_qwen_descartadas = self.repo.auditar_associacoes_qwen()
        self._sugestoes_rapidas_rebaixadas = (
            self.repo.rebaixar_sugestoes_rapidas_antigas()
        )
        # Bancos criados por versões anteriores possuíam apenas a faixa de
        # termos na imagem. Materializa os assentos sem alterar as imagens.
        self._resultados: list[dict] = []
        self._registro_atual: dict | None = None
        self._worker: IndexacaoWorker | None = None
        self._qwen_worker: QwenAreaWorker | None = None
        self._qwen_record_worker: QwenRecordWorker | None = None
        self._qwen_nome_worker: QwenNomeBatchWorker | None = None
        self._import_worker: ImportacaoA07Worker | ImportacaoLivroWorker | None = None
        self._audit_importacao: BookAudit | None = None
        self._sync_worker: SincronizacaoWorker | None = None
        self._reader_window: ImageReaderWindow | None = None
        self._api_server: AcervoApiServer | None = None
        self._processing_dialog: ProcessingDialog | None = None
        self._imagem_exibida = None
        self.setWindowTitle(f"Consulta do Acervo Digitalizado — v{APP_VERSION}")
        self.setMinimumSize(1280, 760)
        self.resize(1500, 900)
        self._init_ui()
        self._iniciar_api()
        self._carregar_filtros()
        self._atualizar_estatisticas()
        self._buscar()
        # A janela fica utilizÃ¡vel imediatamente; a materializaÃ§Ã£o dos
        # registros antigos continua em uma conexÃ£o prÃ³pria em segundo plano.
        QTimer.singleShot(0, self._iniciar_sincronizacao)

    def _iniciar_sincronizacao(self) -> None:
        if self._sync_worker and self._sync_worker.isRunning():
            return
        self._sync_worker = SincronizacaoWorker()
        self._sync_worker.concluido.connect(self._on_sincronizacao_concluida)
        self._sync_worker.erro.connect(self._on_sincronizacao_erro)
        self._sync_worker.finished.connect(self._on_sincronizacao_finalizada)
        self.statusBar().showMessage(
            "Consulta pronta; atualizando registros antigos em segundo plano..."
        )
        self._sync_worker.start()

    @Slot(int)
    def _on_sincronizacao_concluida(self, total: int) -> None:
        self._carregar_filtros()
        self._atualizar_estatisticas()
        self._buscar()
        self.statusBar().showMessage(f"Consulta pronta: {total} registro(s) disponíveis")
        self._auto_resume_processamento()

    @Slot(str)
    def _on_sincronizacao_erro(self, mensagem: str) -> None:
        logger.error("Sincronização de registros: %s", mensagem)
        self.statusBar().showMessage("Consulta aberta; sincronização legada pendente")

    @Slot()
    def _on_sincronizacao_finalizada(self) -> None:
        if self._sync_worker:
            self._sync_worker.deleteLater()
        self._sync_worker = None

    def _init_ui(self) -> None:
        central = QWidget()
        raiz = QVBoxLayout(central)
        raiz.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(central)

        cabecalho = QHBoxLayout()
        titulo = QLabel("CONSULTA DO ACERVO DIGITALIZADO")
        titulo.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        titulo.setStyleSheet("color: #174a7e;")
        cabecalho.addWidget(titulo)
        cabecalho.addStretch()
        importar = QPushButton("Importar livro organizado")
        importar.setToolTip("Audita e importa FRENTE, VERSO, capa, índices e recapturas.")
        importar.clicked.connect(self._importar_livro_organizado)
        cabecalho.addWidget(importar)
        processamento = QPushButton("Processamento")
        processamento.setToolTip("Progresso, histórico, pausa e retomada do OCR por registro.")
        processamento.clicked.connect(self._abrir_processamento)
        cabecalho.addWidget(processamento)
        digitalizador = QPushButton("Abrir Digitalizador")
        digitalizador.clicked.connect(self._abrir_digitalizador)
        cabecalho.addWidget(digitalizador)
        self.lbl_estatisticas = QLabel()
        self.lbl_estatisticas.setStyleSheet("color: #455a64; font-size: 12px;")
        cabecalho.addWidget(self.lbl_estatisticas)
        raiz.addLayout(cabecalho)
        banco = QLabel(f"v{APP_VERSION}  •  Banco: {self.db.path}")
        banco.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        banco.setStyleSheet("color: #607d8b; font-size: 10px;")
        raiz.addWidget(banco)

        filtros = QGroupBox("Pesquisa")
        grade = QGridLayout(filtros)
        self.texto = QLineEdit()
        self.texto.setPlaceholderText("Nome, local, data ou qualquer texto reconhecido...")
        self.texto.returnPressed.connect(self._buscar)
        grade.addWidget(QLabel("Pesquisa livre:"), 0, 0)
        grade.addWidget(self.texto, 0, 1, 1, 5)

        self.termo = QLineEdit()
        self.termo.setValidator(QIntValidator(1, 99_999_999, self))
        self.termo.setPlaceholderText("Ex.: 6802")
        self.termo.returnPressed.connect(self._buscar)
        grade.addWidget(QLabel("Termo:"), 1, 0)
        grade.addWidget(self.termo, 1, 1)

        self.acervo = QComboBox()
        self.oficio = QComboBox()
        self.tipo = QComboBox()
        self.livro = QComboBox()
        grade.addWidget(QLabel("Acervo:"), 1, 2)
        grade.addWidget(self.acervo, 1, 3)
        grade.addWidget(QLabel("Ofício:"), 1, 4)
        grade.addWidget(self.oficio, 1, 5)
        grade.addWidget(QLabel("Tipo:"), 2, 0)
        grade.addWidget(self.tipo, 2, 1)
        grade.addWidget(QLabel("Livro:"), 2, 2)
        grade.addWidget(self.livro, 2, 3, 1, 3)
        self.lbl_livro_status = QLabel("Selecione um livro para ver a cobertura da importacao e dos nomes.")
        self.lbl_livro_status.setStyleSheet("color:#455a64; font-size:11px;")
        grade.addWidget(self.lbl_livro_status, 3, 0, 1, 6)

        botoes = QHBoxLayout()
        self.btn_buscar = QPushButton("BUSCAR")
        self.btn_buscar.setMinimumHeight(36)
        self.btn_buscar.setStyleSheet(
            "QPushButton { background:#1976d2; color:white; font-weight:bold; "
            "border-radius:5px; padding:5px 24px; }"
        )
        self.btn_buscar.clicked.connect(self._buscar)
        botoes.addWidget(self.btn_buscar)
        limpar = QPushButton("Limpar filtros")
        limpar.clicked.connect(self._limpar_filtros)
        botoes.addWidget(limpar)
        exportar = QPushButton("Exportar resultados")
        exportar.clicked.connect(self._exportar_resultados)
        botoes.addWidget(exportar)
        botoes.addStretch()
        self.btn_indexar = QPushButton("Processar somente pendentes")
        self.btn_indexar.setStyleSheet(
            "QPushButton { background:#00897b; color:white; font-weight:bold; "
            "border-radius:5px; padding:5px 16px; }"
        )
        self.btn_indexar.clicked.connect(self._iniciar_indexacao)
        self.btn_indexar.setVisible(False)
        self.btn_qwen_nomes = QPushButton("Qwen: corrigir nomes incertos")
        self.btn_qwen_nomes.setToolTip(
            "Executa o Qwen somente nos nomes que o OCR rápido marcou para revisão."
        )
        self.btn_qwen_nomes.setStyleSheet(
            "QPushButton { background:#6a1b9a; color:white; font-weight:bold; "
            "border-radius:5px; padding:5px 16px; }"
        )
        self.btn_qwen_nomes.clicked.connect(self._iniciar_qwen_nomes_incerto)
        self.btn_qwen_nomes.setVisible(False)
        grade.addLayout(botoes, 4, 0, 1, 6)
        raiz.addWidget(filtros)

        self.progresso = QProgressBar()
        self.progresso.setVisible(False)
        raiz.addWidget(self.progresso)

        principal = QSplitter(Qt.Orientation.Horizontal)
        self.tabela_resultados = QTableWidget(0, 6)
        self.tabela_resultados.setHorizontalHeaderLabels(
            ["Termo", "Nome detectado", "Livro", "Folha", "Face", "Acervo / Ofício"]
        )
        self.tabela_resultados.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tabela_resultados.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tabela_resultados.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tabela_resultados.verticalHeader().setVisible(False)
        self.tabela_resultados.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tabela_resultados.setMinimumWidth(480)
        self.tabela_resultados.itemSelectionChanged.connect(self._mostrar_selecionado)
        principal.addWidget(self.tabela_resultados)

        # A Consulta é deliberadamente leve: não carrega a fotografia nativa
        # nem o OCR completo enquanto o operador apenas percorre resultados.
        # O revisor dedicado é aberto sob demanda, com a imagem original.
        painel_direito = QFrame()
        imagem_layout = QVBoxLayout(painel_direito)
        self.lbl_contexto = QLabel("Selecione um resultado")
        self.lbl_contexto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_contexto.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        imagem_layout.addWidget(self.lbl_contexto)
        self.lbl_status_consulta = QLabel("Selecione um resultado")
        self.lbl_status_consulta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        imagem_layout.addWidget(self.lbl_status_consulta)
        self.thumb_consulta = QLabel("A miniatura aparecerá aqui")
        self.thumb_consulta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_consulta.setMinimumSize(360, 260)
        self.thumb_consulta.setStyleSheet(
            "background:#202124; color:#d9e2ec; border:1px solid #90a4ae;"
        )
        imagem_layout.addWidget(self.thumb_consulta, 1)
        self.lbl_caminho_consulta = QLabel("")
        self.lbl_caminho_consulta.setWordWrap(True)
        self.lbl_caminho_consulta.setStyleSheet("color:#455a64; font-size:11px;")
        imagem_layout.addWidget(self.lbl_caminho_consulta)
        acoes_foto = QHBoxLayout()
        self.btn_revisor = QPushButton("Abrir foto maior / revisor")
        self.btn_revisor.setStyleSheet(
            "QPushButton { background:#1976d2; color:white; font-weight:bold; padding:7px; }"
        )
        self.btn_revisor.clicked.connect(self._abrir_revisor)
        acoes_foto.addWidget(self.btn_revisor)
        abrir_chrome = QPushButton("Abrir no Chrome (orientada)")
        abrir_chrome.setToolTip(
            "Abre uma cópia orientada pela rotação cadastrada; o JPG original não é alterado."
        )
        abrir_chrome.clicked.connect(self._abrir_navegador)
        acoes_foto.addWidget(abrir_chrome)
        abrir_original = QPushButton("Abrir original")
        abrir_original.clicked.connect(self._abrir_original)
        acoes_foto.addWidget(abrir_original)
        acoes_foto.addStretch()
        imagem_layout.addLayout(acoes_foto)
        acoes_rotacao = QHBoxLayout()
        acoes_rotacao.addWidget(QLabel("Orientação da foto:"))
        for texto, delta in (("Girar -90°", -90), ("Girar +90°", 90), ("Girar 180°", 180)):
            botao = QPushButton(texto)
            botao.setToolTip(
                "Salva somente a orientação de visualização. A fotografia original permanece intacta."
            )
            botao.clicked.connect(lambda _checked=False, d=delta: self._girar_imagem(d))
            acoes_rotacao.addWidget(botao)
        zerar_rotacao = QPushButton("Zerar rotação")
        zerar_rotacao.clicked.connect(lambda: self._definir_rotacao(0))
        acoes_rotacao.addWidget(zerar_rotacao)
        acoes_rotacao.addStretch()
        imagem_layout.addLayout(acoes_rotacao)
        acoes_caminho = QHBoxLayout()
        copiar = QPushButton("Copiar caminho")
        copiar.clicked.connect(self._copiar_caminho)
        acoes_caminho.addWidget(copiar)
        copiar_api = QPushButton("Copiar URL da API")
        copiar_api.clicked.connect(self._copiar_url_api)
        acoes_caminho.addWidget(copiar_api)
        abrir_api = QPushButton("Abrir JSON da API")
        abrir_api.clicked.connect(self._abrir_json_api)
        acoes_caminho.addWidget(abrir_api)
        acoes_caminho.addStretch()
        imagem_layout.addLayout(acoes_caminho)
        legenda = QLabel(
            "<b>Legenda:</b> "
            "<span style='color:#c62828'>● vermelho = revisar</span> &nbsp; "
            "<span style='color:#2e7d32'>● verde = confirmado</span> &nbsp; "
            "<span style='color:#1565c0'>● azul = informação/OCR</span>"
        )
        legenda.setTextFormat(Qt.TextFormat.RichText)
        imagem_layout.addWidget(legenda)
        principal.addWidget(painel_direito)
        principal.setStretchFactor(0, 2)
        principal.setStretchFactor(1, 5)
        principal.setSizes([520, 900])
        raiz.addWidget(principal, 1)

        self.statusBar().showMessage("Pronto")

    @staticmethod
    def _id_combo(combo: QComboBox) -> int | None:
        valor = combo.currentData()
        return int(valor) if valor is not None else None

    def _carregar_filtros(self) -> None:
        def preencher(combo: QComboBox, itens: list[dict], campo: str) -> None:
            combo.clear()
            combo.addItem("Todos", None)
            for item in itens:
                combo.addItem(str(item.get(campo) or ""), item["id"])

        preencher(self.acervo, self.repo.listar_acervos(), "nome")
        preencher(self.oficio, self.repo.listar_oficios(), "nome")
        preencher(self.tipo, self.repo.listar_tipos(), "nome")
        self._carregar_livros()
        for combo in (self.acervo, self.oficio, self.tipo):
            combo.currentIndexChanged.connect(self._carregar_livros)
        self.livro.currentIndexChanged.connect(self._atualizar_resumo_livro)

    @Slot()
    def _carregar_livros(self) -> None:
        atual = self.livro.currentData() if self.livro.count() else None
        self.livro.clear()
        self.livro.addItem("Todos os livros", None)
        acervo_id = self._id_combo(self.acervo)
        oficio_id = self._id_combo(self.oficio)
        tipo_id = self._id_combo(self.tipo)
        for livro in self.repo.listar_acervo_livros():
            if acervo_id is not None and livro.get("acervo_id") != acervo_id:
                continue
            if oficio_id is not None and livro.get("oficio_id") != oficio_id:
                continue
            if tipo_id is not None and livro.get("tipo_id") != tipo_id:
                continue
            nome = (
                f"{livro.get('codigo') or '?'} - {livro.get('tipo_nome') or ''} "
                f"({livro.get('total_registros', 0)}/{livro.get('total_esperado', 0)} registros)"
            )
            self.livro.addItem(nome, livro["id"])
        if atual is not None:
            indice = self.livro.findData(atual)
            if indice >= 0:
                self.livro.setCurrentIndex(indice)
        self._atualizar_resumo_livro()

    @Slot()
    def _atualizar_resumo_livro(self) -> None:
        livro_id = self._id_combo(self.livro)
        if livro_id is None:
            self.lbl_livro_status.setText(
                "Livros cadastrados permanecem visiveis mesmo antes de receber imagens."
            )
            return
        livro = next(
            (item for item in self.repo.listar_acervo_livros() if int(item["id"]) == livro_id),
            None,
        )
        if not livro:
            return
        esperado = int(livro.get("total_esperado") or 0)
        importado = int(livro.get("total_registros") or 0)
        self.lbl_livro_status.setText(
            f"Esperado: {esperado}  |  Importado: {importado}  |  "
            f"Faltante: {max(0, esperado-importado)} ({int(livro.get('faces_faltantes') or 0)} face(s))  |  "
            f"Nomes: {int(livro.get('nomes_processados') or 0)} processados, "
            f"{int(livro.get('nomes_pendentes') or 0)} pendentes, "
            f"{int(livro.get('nomes_sugestoes') or 0)} sugestoes, "
            f"{int(livro.get('nomes_revisao') or 0)} revisar, "
            f"{int(livro.get('nomes_confirmados') or 0)} confirmados."
        )

    @Slot()
    def _buscar(self) -> None:
        termo = int(self.termo.text()) if self.termo.text().strip() else None
        self._resultados = self.repo.buscar_registros(
            texto=self.texto.text(),
            termo=termo,
            acervo_id=self._id_combo(self.acervo),
            oficio_id=self._id_combo(self.oficio),
            tipo_id=self._id_combo(self.tipo),
            livro_id=self._id_combo(self.livro),
        )
        self.tabela_resultados.setRowCount(len(self._resultados))
        for linha, resultado in enumerate(self._resultados):
            nome_exibido = resultado.get("nome_confirmado") or resultado.get("nome_sugerido") or "—"
            if resultado.get("nome_sugerido") and not resultado.get("nome_confirmado"):
                nome_exibido = (
                    f"{nome_exibido} · sugestão "
                    f"{float(resultado.get('nome_confianca') or 0) * 100:.0f}%"
                )
            valores = [
                resultado.get("termo") or "?",
                nome_exibido,
                resultado.get("livro_codigo") or "?",
                resultado.get("folha") or "?",
                (resultado.get("face") or "?").capitalize(),
                f"{resultado.get('acervo_nome') or '?'} / {resultado.get('oficio_nome') or '?'}",
            ]
            for coluna, valor in enumerate(valores):
                item = QTableWidgetItem(str(valor))
                if coluna == 0:
                    item.setData(Qt.ItemDataRole.UserRole, resultado["registro_id"])
                self.tabela_resultados.setItem(linha, coluna, item)
            # Cores são status operacional, não decoração: a lista pode ser
            # conferida rapidamente sem abrir cada fotografia.
            estado = estado_visual_registro(
                resultado, self.repo.tem_revisao_pendente(resultado["imagem_id"])
            )
            cor = QColor(estado["fundo"])
            texto_cor = QColor(estado["frente"])
            for coluna in range(self.tabela_resultados.columnCount()):
                item = self.tabela_resultados.item(linha, coluna)
                if item is not None:
                    item.setBackground(cor)
                    item.setForeground(texto_cor)
        self.tabela_resultados.resizeColumnsToContents()
        self.tabela_resultados.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.statusBar().showMessage(f"{len(self._resultados)} registro(s) encontrado(s)")
        if self._resultados:
            self.tabela_resultados.selectRow(0)
        else:
            self._registro_atual = None
            self._imagem_exibida = None
            self.thumb_consulta.clear()
            self.lbl_caminho_consulta.clear()
            self.lbl_contexto.setText("Nenhum registro encontrado")
            self.lbl_status_consulta.setText("Nenhum registro encontrado")

    def _limpar_filtros(self) -> None:
        self.texto.clear()
        self.termo.clear()
        self.acervo.setCurrentIndex(0)
        self.oficio.setCurrentIndex(0)
        self.tipo.setCurrentIndex(0)
        self.livro.setCurrentIndex(0)
        self._buscar()

    def _exportar_resultados(self) -> None:
        if not self._resultados:
            QMessageBox.information(self, "Exportar", "Não há resultados para exportar.")
            return
        path, filtro = QFileDialog.getSaveFileName(
            self,
            "Exportar consulta e metadados",
            "consulta_acervo.json",
            "JSON estruturado (*.json);;Planilha CSV (*.csv)",
        )
        if not path:
            return
        saida = []
        for registro in self._resultados:
            item = dict(registro)
            item["metadados"] = self.repo.listar_metadados_registro(
                registro["registro_id"]
            )
            saida.append(item)
        try:
            if path.lower().endswith(".csv") or "CSV" in filtro:
                campos = [
                    "acervo", "oficio", "tipo", "livro", "termo", "folha",
                    "face", "nome_registrado", "nome_confirmado", "nome_sugerido",
                    "nome_status", "nome_confianca", "nome_fonte", "imagem", "metadados_json",
                ]
                with open(path, "w", newline="", encoding="utf-8-sig") as arquivo:
                    writer = csv.DictWriter(arquivo, fieldnames=campos)
                    writer.writeheader()
                    for registro in saida:
                        writer.writerow({
                            "acervo": registro.get("acervo_nome"),
                            "oficio": registro.get("oficio_nome"),
                            "tipo": registro.get("tipo_nome"),
                            "livro": registro.get("livro_codigo"),
                            "termo": registro.get("termo"),
                            "folha": registro.get("folha"),
                            "face": registro.get("face"),
                            "nome_registrado": registro.get("nomes"),
                            "nome_confirmado": registro.get("nome_confirmado"),
                            "nome_sugerido": registro.get("nome_sugerido"),
                            "nome_status": registro.get("nome_status"),
                            "nome_confianca": registro.get("nome_confianca"),
                            "nome_fonte": registro.get("nome_fonte"),
                            "imagem": registro.get("caminho_original"),
                            "metadados_json": json.dumps(
                                registro["metadados"], ensure_ascii=False
                            ),
                        })
            else:
                if not path.lower().endswith(".json"):
                    path += ".json"
                Path(path).write_text(
                    json.dumps(saida, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            self.statusBar().showMessage(f"Resultados exportados: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Falha ao exportar", str(exc))

    @Slot()
    def _mostrar_selecionado(self) -> None:
        linha = self.tabela_resultados.currentRow()
        if linha < 0 or linha >= len(self._resultados):
            return
        self._registro_atual = self._resultados[linha]
        registro = self._registro_atual
        contexto = (
            f"Termo {registro.get('termo') or '?'} | Livro {registro.get('livro_codigo') or '?'} | "
            f"Folha {registro.get('folha') or '?'} - {(registro.get('face') or '?').capitalize()}"
        )
        evidencia = texto_evidencia_termo(registro)
        if evidencia:
            contexto = f"{contexto}\n{evidencia}"
        self.lbl_contexto.setText(contexto)
        self._reexibir_imagem()
        self.lbl_status_consulta.setText(self._status_consulta(registro))
        estado = estado_visual_registro(
            registro, self.repo.tem_revisao_pendente(registro["imagem_id"])
        )
        self.lbl_status_consulta.setStyleSheet(
            f"color:{estado['frente']}; font-weight:bold;"
        )

    def _status_consulta(self, registro: dict) -> str:
        return estado_visual_registro(
            registro, self.repo.tem_revisao_pendente(registro["imagem_id"])
        )["texto"]

    @Slot()
    def _reexibir_imagem(self) -> None:
        if not self._registro_atual:
            return
        r = self._registro_atual
        path = Path(r.get("caminho_original") or "")
        if not path.is_file():
            self.thumb_consulta.clear()
            self.lbl_contexto.setText(self.lbl_contexto.text() + " | imagem não encontrada")
            return
        thumb_path = Path(r.get("caminho_thumb") or path)
        if not thumb_path.is_file():
            thumb_path = path
        dados = np.fromfile(str(thumb_path), dtype=np.uint8)
        bgr = cv2.imdecode(dados, cv2.IMREAD_COLOR)
        if bgr is None:
            self.thumb_consulta.clear()
            self.lbl_caminho_consulta.setText(str(path))
            return
        bgr = aplicar_rotacao(bgr, int(r.get("rotacao_visualizacao") or 0))
        retificada = retificar_formulario(bgr)
        if retificada.applied:
            bgr = retificada.image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.thumb_consulta.setPixmap(
            pixmap.scaled(
                self.thumb_consulta.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.lbl_caminho_consulta.setText(str(path))
        self._imagem_exibida = None

    def _criar_contexto_qwen(
        self,
        registro: dict,
        bbox,
        *,
        tipo: str,
    ) -> QwenJobContext | None:
        total = len(self.repo.listar_registros_imagem(int(registro["imagem_id"]))) or 1
        indice = int(registro.get("indice_na_imagem") or 0)
        valido = (
            bbox_corresponde_registro(bbox, indice, total)
            if tipo == "registro"
            else bbox_contido_no_registro(bbox, indice, total)
        )
        if not valido:
            QMessageBox.warning(
                self,
                "Área incompatível",
                "A seleção não pertence integralmente ao assento aberto. "
                "Reposicione a moldura antes de executar o Qwen.",
            )
            return None
        return QwenJobContext.from_registro(
            registro, bbox, total=total, tipo=tipo
        )

    def _registro_do_contexto(self, contexto: QwenJobContext) -> dict | None:
        return validar_contexto_qwen(self.repo, contexto)

    def _iniciar_qwen_registro(self, bbox=None, image=None, registro_contexto=None) -> None:
        if self._qwen_record_worker and self._qwen_record_worker.isRunning():
            return
        if self._qwen_worker and self._qwen_worker.isRunning():
            QMessageBox.information(self, "Qwen", "Aguarde a leitura atual terminar.")
            return
        registro = dict(registro_contexto or self._registro_atual or {})
        if not registro or image is None or bbox is None:
            return
        contexto = self._criar_contexto_qwen(registro, bbox, tipo="registro")
        if contexto is None:
            return
        existente = self.repo.db.fetchone(
            """
            SELECT id FROM ocr_deteccao
            WHERE registro_id=? AND fonte='qwen_registro' AND ativo=1
            LIMIT 1
            """,
            (contexto.registro_id,),
        )
        if existente:
            QMessageBox.information(
                self,
                "Registro já processado",
                "Os três campos básicos já foram lidos e estão salvos. "
                "Confira/corrija-os no painel do revisor.",
            )
            return
        model_path = self.settings.get("ocr", "qwen_model_path", "") or None
        if not modelo_qwen_instalado(model_path):
            resposta = QMessageBox.question(
                self,
                "Baixar Qwen2-VL 2B",
                f"O modelo oficial ocupa aproximadamente {QWEN_MODEL_SIZE_MIB / 1024:.1f} GB. "
                "Deseja baixar agora?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resposta != QMessageBox.StandardButton.Yes:
                return
        self._qwen_record_worker = QwenRecordWorker(
            image,
            contexto,
            model_path,
            int(self.settings.get("ocr", "qwen_max_new_tokens", 96)),
        )
        self._qwen_record_worker.concluido.connect(self._on_qwen_registro_concluido)
        self._qwen_record_worker.erro.connect(self._on_qwen_erro)
        self._qwen_record_worker.finished.connect(self._on_qwen_registro_finalizado)
        self.progresso.setVisible(True)
        self.progresso.setRange(0, 0)
        self.progresso.setFormat("Qwen lendo nome, termo, mãe e data — uma única chamada...")
        self._qwen_record_worker.start()

    @Slot(dict)
    def _on_qwen_registro_concluido(self, resultado: dict) -> None:
        dados_contexto = resultado.get("job_context") or {}
        try:
            contexto = QwenJobContext(**{
                **dados_contexto,
                "bbox": tuple(dados_contexto.get("bbox") or ()),
            })
        except (TypeError, ValueError):
            return
        registro_atual = self._registro_do_contexto(contexto)
        if registro_atual is None:
            QMessageBox.warning(
                self,
                "Resultado descartado",
                "O registro, a fotografia ou a região mudou durante a leitura. "
                "O resultado não foi associado a outro termo.",
            )
            return
        registro = {
            "registro_id": contexto.registro_id,
            "imagem_id": contexto.imagem_id,
            "termo": contexto.termo,
        }
        campos = resultado.get("campos") or {}
        texto_bruto = resultado.get("texto_bruto") or json.dumps(campos, ensure_ascii=False)
        execucao_id = self.repo.criar_execucao_ocr(
            imagem_id=registro["imagem_id"],
            registro_id=registro["registro_id"],
            motor=resultado.get("motor") or "qwen2-vl-2b-registro",
            texto_bruto=texto_bruto,
            tempo_ms=resultado.get("tempo_ms") or 0,
            sucesso=True,
            substituir_ativa=False,
        )
        bbox_json = json.dumps(resultado.get("bbox_relativo"), ensure_ascii=False)
        deteccoes = []
        for tipo, valor in (
            ("nome_registrado", campos.get("nome_registrado")),
            ("termo", campos.get("termo")),
            ("nome_mae", campos.get("nome_mae")),
            ("data", campos.get("data_registro")),
        ):
            valor_bruto = tratar_valor(str(valor or ""))
            contexto_extra = ""
            if tipo == "termo":
                valor, contexto_extra = _tratar_termo_qwen(
                    valor_bruto, registro.get("termo")
                )
            else:
                valor = valor_bruto
            if not valor:
                continue
            deteccoes.append({
                "tipo": tipo,
                "valor_original": valor,
                "valor_tratado": valor,
                "valor_normalizado": normalizar_busca(valor),
                "confianca": 0.45,
                "motor": resultado.get("motor") or "qwen2-vl-2b-registro",
                "fonte": "qwen_registro",
                "status": "precisa_revisao",
                "bbox_json": bbox_json,
                "contexto": (
                    "Leitura estruturada de uma única área de registro; confirmar no revisor. "
                    + contexto_extra
                ).strip(),
            })
        if deteccoes:
            self.repo.salvar_deteccoes_ocr(
                execucao_id=execucao_id,
                imagem_id=registro["imagem_id"],
                registro_id=registro["registro_id"],
                deteccoes=deteccoes,
            )
        if self._registro_atual and int(self._registro_atual["registro_id"]) == contexto.registro_id:
            self._carregar_metadados()
        self._buscar()
        self.statusBar().showMessage(
            f"Qwen leu {len(deteccoes)} campo(s) em {(resultado.get('tempo_ms') or 0) / 1000:.0f}s; revisar no painel."
        )

    @Slot()
    def _on_qwen_registro_finalizado(self) -> None:
        if self._qwen_record_worker:
            self._qwen_record_worker.deleteLater()
        self._qwen_record_worker = None
        self.progresso.setRange(0, 1)
        self.progresso.setVisible(False)

    def _iniciar_qwen_area(self, bbox=None, image=None, registro_contexto=None) -> None:
        if self._qwen_worker and self._qwen_worker.isRunning():
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self, "Qwen", "Aguarde a indexação em lote terminar."
            )
            return
        registro = dict(registro_contexto or self._registro_atual or {})
        if not registro:
            return
        source_image = image if image is not None else self._imagem_exibida
        if source_image is None or bbox is None:
            QMessageBox.information(
                self,
                "Selecionar área",
                "Abra o revisor, ative 'Selecionar área' e arraste sobre a linha.",
            )
            return
        contexto = self._criar_contexto_qwen(registro, bbox, tipo="area")
        if contexto is None:
            return
        rotulo, ok = QInputDialog.getItem(
            self,
            "Conteúdo da área",
            "O que deve ser lido?",
            list(PROMPTS_QWEN),
            0,
            False,
        )
        if not ok:
            return
        tipo, instrucao = PROMPTS_QWEN[rotulo]
        for existente in self.repo.listar_deteccoes_area(
            imagem_id=contexto.imagem_id,
            registro_id=contexto.registro_id,
            tipo=tipo,
        ):
            try:
                area_salva = json.loads(existente.get("bbox_json") or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if _mesma_area(area_salva, bbox):
                QMessageBox.information(
                    self,
                    "Área já processada",
                    "Esta área já foi lida uma vez. O resultado salvo é:\n\n"
                    f"{existente.get('valor_tratado') or existente.get('valor_original') or '(sem texto)'}\n\n"
                    "Use a tabela de metadados para confirmar ou corrigir; o Qwen não será executado novamente.",
                )
                return
        model_path = self.settings.get("ocr", "qwen_model_path", "") or None
        if not modelo_qwen_instalado(model_path):
            resposta = QMessageBox.question(
                self,
                "Baixar Qwen2-VL 2B",
                f"O modelo oficial ocupa aproximadamente {QWEN_MODEL_SIZE_MIB / 1024:.1f} GB. "
                "Deseja baixar agora?\n\nDepois da instalação a leitura funciona localmente.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resposta != QMessageBox.StandardButton.Yes:
                return

        altura, largura = source_image.shape[:2]
        x1 = max(0, min(largura - 1, int(bbox[0] * largura)))
        y1 = max(0, min(altura - 1, int(bbox[1] * altura)))
        x2 = max(x1 + 1, min(largura, int(bbox[2] * largura)))
        y2 = max(y1 + 1, min(altura, int(bbox[3] * altura)))
        recorte = source_image[y1:y2, x1:x2]
        if recorte.shape[0] < 12 or recorte.shape[1] < 24:
            QMessageBox.information(self, "Selecionar área", "A área selecionada é pequena demais.")
            return

        self._qwen_worker = QwenAreaWorker(
            recorte,
            tipo,
            instrucao,
            bbox,
            model_path,
            int(self.settings.get("ocr", "qwen_max_new_tokens", 96)),
            contexto,
            bool(self.settings.get("ocr", "qwen_preprocess", False)),
        )
        self._qwen_worker.concluido.connect(self._on_qwen_concluido)
        self._qwen_worker.erro.connect(self._on_qwen_erro)
        self._qwen_worker.finished.connect(self._on_qwen_finalizado)
        self.progresso.setVisible(True)
        self.progresso.setRange(0, 0)
        self.progresso.setFormat("Qwen lendo a área na CPU — estimativa: 2 a 4 minutos...")
        self._qwen_worker.start()

    @Slot(dict)
    def _on_qwen_concluido(self, resultado: dict) -> None:
        dados_contexto = resultado.get("job_context") or {}
        try:
            contexto = QwenJobContext(**{
                **dados_contexto,
                "bbox": tuple(dados_contexto.get("bbox") or ()),
            })
        except (TypeError, ValueError):
            return
        if self._registro_do_contexto(contexto) is None:
            QMessageBox.warning(
                self,
                "Resultado descartado",
                "A fotografia ou o assento mudou durante a leitura. O resultado "
                "não foi salvo em outro termo.",
            )
            return
        texto = tratar_valor(resultado["texto"])
        registro = {
            "registro_id": contexto.registro_id,
            "imagem_id": contexto.imagem_id,
            "termo": contexto.termo,
        }
        execucao_id = self.repo.criar_execucao_ocr(
            imagem_id=registro["imagem_id"],
            registro_id=registro["registro_id"],
            motor=resultado["motor"],
            texto_bruto=texto,
            tempo_ms=resultado["tempo_ms"],
            sucesso=True,
            substituir_ativa=False,
        )
        bbox_json = json.dumps(resultado["bbox_relativo"])
        self.repo.salvar_deteccoes_ocr(
            execucao_id=execucao_id,
            imagem_id=registro["imagem_id"],
            registro_id=registro["registro_id"],
            deteccoes=[{
                "tipo": resultado["tipo"],
                "valor_original": texto,
                "valor_tratado": texto,
                "valor_normalizado": normalizar_busca(texto),
                "confianca": 0.45,
                "motor": resultado["motor"],
                "fonte": "qwen_area",
                "status": "precisa_revisao",
                "bbox_json": bbox_json,
                "contexto": "Área escolhida manualmente pelo operador",
            }],
        )
        confirmado, ok = QInputDialog.getText(
            self,
            "Revisar sugestão do Qwen",
            "Confira na imagem, corrija se necessário e confirme:",
            text=texto,
        )
        if ok and confirmado.strip():
            self.repo.salvar_metadado_tratado(
                imagem_id=registro["imagem_id"],
                registro_id=registro["registro_id"],
                tipo=resultado["tipo"],
                valor=confirmado.strip(),
                confianca=1.0,
                fonte="qwen_area_confirmado",
                motor="operador+qwen2-vl",
                status="confirmado",
                contexto=f"Sugestão original: {texto}",
            )
        if self._registro_atual and int(self._registro_atual["registro_id"]) == contexto.registro_id:
            self._carregar_metadados()
        self._buscar()
        self.statusBar().showMessage(
            f"Qwen concluiu em {resultado['tempo_ms'] / 1000:.0f} s; sugestão preservada."
        )

    @Slot(str)
    def _on_qwen_erro(self, mensagem: str) -> None:
        QMessageBox.warning(self, "Falha no Qwen", mensagem)

    @Slot()
    def _on_qwen_finalizado(self) -> None:
        if self._qwen_worker:
            self._qwen_worker.deleteLater()
        self._qwen_worker = None
        self.progresso.setRange(0, 1)
        self.progresso.setVisible(False)

    def _carregar_metadados(self) -> None:
        if not self._registro_atual:
            return
        registro_id = self._registro_atual["registro_id"]
        metadados = self.repo.listar_metadados_registro(registro_id)
        if self._reader_window is not None:
            self._reader_window.atualizar_metadados(metadados)
            return
        self.tabela_metadados.setRowCount(len(metadados))
        for linha, meta in enumerate(metadados):
            valores = [
                TIPOS_METADADO.get(meta.get("tipo"), meta.get("tipo") or "?"),
                meta.get("valor_tratado") or meta.get("valor_original") or "",
                f"{float(meta.get('confianca') or 0) * 100:.0f}%",
                meta.get("motor") or meta.get("fonte") or "—",
                meta.get("status") or "—",
                meta.get("escopo") or "—",
            ]
            for coluna, valor in enumerate(valores):
                item = QTableWidgetItem(str(valor))
                if coluna == 0:
                    item.setData(Qt.ItemDataRole.UserRole, meta["id"])
                self.tabela_metadados.setItem(linha, coluna, item)
        execucoes = self.repo.listar_execucoes_registro(registro_id)
        partes = []
        for execucao in execucoes:
            partes.append(
                f"[{execucao.get('motor')} | {execucao.get('escopo')} | "
                f"{float(execucao.get('tempo_ms') or 0):.0f} ms]\n"
                f"{execucao.get('texto_bruto') or '(sem texto)'}"
            )
        self.texto_ocr.setPlainText("\n\n".join(partes) if partes else "OCR ainda não indexado para este assento.")

    def _deteccao_selecionada(self) -> int | None:
        tabela = self._reader_window.tabela_metadados if self._reader_window is not None else getattr(self, "tabela_metadados", None)
        if tabela is None:
            return None
        linha = tabela.currentRow()
        if linha < 0:
            return None
        item = tabela.item(linha, 0)
        return int(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _adicionar_metadado(self) -> None:
        if self._reader_window is not None:
            self._reader_window._adicionar_metadado()
            return
        if not self._registro_atual:
            return
        nomes = list(TIPOS_METADADO.values())[:-1]
        rotulo, ok = QInputDialog.getItem(self, "Campo", "Tipo do metadado:", nomes, 0, False)
        if not ok:
            return
        valor, ok = QInputDialog.getText(self, "Valor", f"{rotulo}:")
        if not ok or not valor.strip():
            return
        tipo = next(chave for chave, nome in TIPOS_METADADO.items() if nome == rotulo)
        self.repo.salvar_metadado_tratado(
            imagem_id=self._registro_atual["imagem_id"],
            registro_id=self._registro_atual["registro_id"],
            tipo=tipo,
            valor=valor.strip(),
            confianca=1.0,
            fonte="operador",
            motor="operador",
            status="confirmado",
        )
        self._carregar_metadados()
        self._buscar()

    def _corrigir_metadado(self) -> None:
        if self._reader_window is not None:
            self._reader_window._corrigir_metadado()
            return
        deteccao_id = self._deteccao_selecionada()
        linha = self.tabela_metadados.currentRow()
        if deteccao_id is None or linha < 0:
            return
        atual = self.tabela_metadados.item(linha, 1).text()
        novo, ok = QInputDialog.getText(self, "Corrigir metadado", "Valor correto:", text=atual)
        if ok and novo.strip() and novo.strip() != atual:
            self.repo.corrigir_deteccao(deteccao_id, novo.strip())
            self._carregar_metadados()
            self._buscar()

    def _confirmar_metadado(self) -> None:
        if self._reader_window is not None:
            self._reader_window._confirmar_metadado()
            return
        deteccao_id = self._deteccao_selecionada()
        if deteccao_id is not None:
            self.repo.confirmar_deteccao(deteccao_id)
            self._carregar_metadados()

    def _carregar_imagem_original(self, path: Path):
        """Lê o JPG por bytes para funcionar também com pastas acentuadas."""
        dados = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None

    def _garantir_dialogo_processamento(self) -> ProcessingDialog:
        if self._processing_dialog is None:
            self._processing_dialog = ProcessingDialog(
                repo=self.repo,
                settings=self.settings,
                abrir_registro=self._abrir_registro_por_id,
                parent=self,
            )
        return self._processing_dialog

    @Slot()
    def _abrir_processamento(self) -> None:
        dialogo = self._garantir_dialogo_processamento()
        dialogo.atualizar()
        dialogo.show()
        dialogo.raise_()
        dialogo.activateWindow()

    def _auto_resume_processamento(self) -> None:
        lote = self.repo.db.fetchone(
            """
            SELECT id FROM processamento_lote
            WHERE status='processando' ORDER BY updated_at DESC LIMIT 1
            """
        )
        if lote:
            self._garantir_dialogo_processamento().auto_resume()

    def _abrir_registro_por_id(self, registro_id: int) -> None:
        base = self.repo.get_registro(int(registro_id))
        if not base:
            return
        resultados = self.repo.buscar_registros(
            termo=base.get("termo"), livro_id=base.get("livro_id"), limite=20
        )
        registro = next(
            (item for item in resultados if int(item["registro_id"]) == int(registro_id)),
            None,
        )
        if registro is None:
            return
        self._registro_atual = registro
        self._abrir_revisor()

    def _abrir_revisor(self) -> None:
        if not self._registro_atual:
            return
        path = Path(self._registro_atual.get("caminho_original") or "")
        if not path.is_file():
            QMessageBox.warning(self, "Imagem", "O arquivo original não foi encontrado.")
            return
        image = self._carregar_imagem_original(path)
        if image is None:
            QMessageBox.warning(self, "Imagem", "Não foi possível decodificar a fotografia.")
            return
        rotacao = int(self._registro_atual.get("rotacao_visualizacao") or 0)
        image = aplicar_rotacao(image, rotacao)
        # A exibição usa a mesma retificação do OCR: a foto fica "enquadrada"
        # e a moldura do assento passa a bater com as linhas reais do formulário.
        # applied=True garante ao menos o alinhamento horizontal (nunca uma
        # transformação pior); applied=False devolve a cópia sem mudança.
        retificada = retificar_formulario(image)
        if retificada.applied:
            image = retificada.image
        self._imagem_exibida = image
        total = max(
            1,
            int(
                (self._registro_atual.get("termo_final") or self._registro_atual.get("termo") or 0)
                - (self._registro_atual.get("termo_inicial") or self._registro_atual.get("termo") or 0)
                + 1
            ),
        )
        if self._reader_window is not None:
            self._reader_window.close()
        self._reader_window = ImageReaderWindow(
            image_path=path,
            image=image,
            registro=self._registro_atual,
            metadados=self.repo.listar_metadados_registro(self._registro_atual["registro_id"]),
            repo=self.repo,
            parent=self,
        )
        registro_aberto = dict(self._registro_atual)
        self._reader_window.qwen_requested.connect(
            lambda imagem, bbox, registro=registro_aberto: self._iniciar_qwen_area(
                bbox, imagem, registro
            )
        )
        self._reader_window.qwen_record_requested.connect(
            lambda imagem, bbox, registro=registro_aberto: self._iniciar_qwen_registro(
                bbox, imagem, registro
            )
        )
        self._reader_window.show()
        self._reader_window.raise_()
        self._reader_window.activateWindow()
        self.statusBar().showMessage(
            f"Revisor aberto em alta resolução — registro {int(self._registro_atual.get('indice_na_imagem') or 0) + 1}/{total}"
        )

    @staticmethod
    def _chrome_executavel() -> Path | None:
        candidatos = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
        return next((c for c in candidatos if c.is_file()), None)

    def _abrir_url_navegador(self, url: str) -> None:
        chrome = self._chrome_executavel()
        if chrome is not None:
            ok, _pid = QProcess.startDetached(str(chrome), ["--new-tab", url])
            if ok:
                return
        QDesktopServices.openUrl(QUrl(url))

    def _abrir_navegador(self) -> None:
        if not self._registro_atual:
            return
        path = Path(self._registro_atual.get("caminho_original") or "")
        if path.is_file():
            try:
                orientada = materializar_copia_orientada(
                    path,
                    self._registro_atual.get("rotacao_visualizacao") or 0,
                )
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Imagem", f"Não foi possível preparar a foto orientada: {exc}")
                return
            self._abrir_url_navegador(orientada.as_uri())

    def _definir_rotacao(self, rotacao: int) -> None:
        if not self._registro_atual:
            return
        imagem_id = self._registro_atual.get("imagem_id")
        if imagem_id is None:
            return
        valor = normalizar_rotacao(rotacao)
        self.repo.atualizar_imagem(int(imagem_id), rotacao_visualizacao=valor)
        self._registro_atual["rotacao_visualizacao"] = valor
        for registro in self._resultados:
            if int(registro.get("imagem_id") or -1) == int(imagem_id):
                registro["rotacao_visualizacao"] = valor
        self._reexibir_imagem()
        if self._reader_window is not None:
            self._reader_window.close()
            self._reader_window = None
            self._abrir_revisor()
        self.statusBar().showMessage(f"Orientação salva: {valor}° (original preservado)")

    def _girar_imagem(self, delta: int) -> None:
        if not self._registro_atual:
            return
        atual = int(self._registro_atual.get("rotacao_visualizacao") or 0)
        self._definir_rotacao(atual + int(delta))

    def _copiar_caminho(self) -> None:
        if not self._registro_atual:
            return
        QApplication.clipboard().setText(str(Path(self._registro_atual.get("caminho_original") or "").resolve()))
        self.statusBar().showMessage("Caminho completo copiado")

    def _copiar_url_api(self) -> None:
        if not self._registro_atual or self._api_server is None or not self._api_server.running:
            QMessageBox.information(self, "API", "A API local não está disponível.")
            return
        url = f"{self._api_server.base_url}/api/v1/registros/{self._registro_atual['registro_id']}"
        QApplication.clipboard().setText(url)
        self.statusBar().showMessage(f"URL da API copiada: {url}")

    def _abrir_json_api(self) -> None:
        if not self._registro_atual or self._api_server is None or not self._api_server.running:
            return
        url = f"{self._api_server.base_url}/api/v1/registros/{self._registro_atual['registro_id']}"
        self._abrir_url_navegador(url)

    def _iniciar_api(self) -> None:
        host = str(self.settings.get("api", "host", "127.0.0.1"))
        port = int(self.settings.get("api", "port", 8765))
        self._api_server = AcervoApiServer(self.repo, host=host, port=port)
        try:
            self._api_server.start()
            self.statusBar().showMessage(
                f"Pronto — API local disponível em {self._api_server.base_url}/api/v1"
            )
        except OSError as exc:
            self._api_server = None
            self.statusBar().showMessage(f"Consulta pronta (API local indisponível: {exc})")

    def _abrir_original(self) -> None:
        if not self._registro_atual:
            return
        path = Path(self._registro_atual.get("caminho_original") or "")
        if path.is_file():
            try:
                os.startfile(str(path))
            except (AttributeError, OSError):
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _iniciar_indexacao(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        livro_id = self._id_combo(self.livro)
        imagens = self.repo.listar_imagens_para_indexacao(
            livro_id=livro_id,
            somente_pendentes=True,
        )
        if not imagens:
            QMessageBox.information(self, "Indexação", "Não há imagens pendentes neste filtro.")
            return
        escopo = self.livro.currentText() if livro_id else "todos os livros"
        resposta = QMessageBox.question(
            self,
            "Processar OCR pendente",
            f"Processar {len(imagens)} fotografia(s) ainda pendente(s) de {escopo}?\n\n"
            "Cada fotografia será processada uma única vez. Resultados e falhas "
            "ficarão salvos junto ao cadastro da imagem.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if resposta != QMessageBox.StandardButton.Yes:
            return
        self._worker = IndexacaoWorker(
            self.repo,
            self.settings,
            imagens,
        )
        self._worker.progresso.connect(self._on_progresso_indexacao)
        self._worker.erro.connect(self._on_erro_indexacao)
        self._worker.concluido.connect(self._on_indexacao_concluida)
        self._worker.finished.connect(self._on_worker_finalizado)
        self.btn_indexar.setEnabled(False)
        self.progresso.setVisible(True)
        self.progresso.setRange(0, len(imagens))
        self._worker.start()

    def _listar_nomes_incerto(self) -> list[dict]:
        return self.repo.db.fetchall(
            """
            SELECT DISTINCT
                   r.id AS registro_id, r.termo, r.indice_na_imagem,
                   i.id AS imagem_id, i.caminho_original,
                   i.termo_inicial, i.termo_final,
                   i.livro_id
            FROM ocr_deteccao d
            JOIN registro r ON r.id=d.registro_id
            JOIN imagem i ON i.id=d.imagem_id
            WHERE d.ativo=1 AND d.fonte='ocr_nome_rapido'
              AND d.confianca < ?
              AND NOT EXISTS (
                  SELECT 1 FROM ocr_deteccao q
                  WHERE q.ativo=1 AND q.registro_id=r.id
                    AND q.tipo='nome_registrado'
                    AND q.fonte='qwen_nome_correcao'
              )
            ORDER BY i.livro_id, r.termo, r.id
            """,
            (float(self.settings.get("ocr", "name_qwen_threshold", 0.78)),),
        )

    def _iniciar_qwen_nomes_incerto(self) -> None:
        if self._qwen_nome_worker and self._qwen_nome_worker.isRunning():
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Qwen", "Aguarde a indexação rápida terminar.")
            return
        itens = self._listar_nomes_incerto()
        if not itens:
            QMessageBox.information(
                self,
                "Qwen — nomes incertos",
                "Não há nomes abaixo do limiar aguardando correção.",
            )
            return
        model_path = self.settings.get("ocr", "qwen_model_path", "") or None
        if not modelo_qwen_instalado(model_path):
            resposta = QMessageBox.question(
                self,
                "Baixar Qwen2-VL 2B",
                f"O modelo ocupa aproximadamente {QWEN_MODEL_SIZE_MIB / 1024:.1f} GB. "
                f"A fila contém {len(itens)} nome(s) incerto(s). Baixar agora?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resposta != QMessageBox.StandardButton.Yes:
                return
        self._qwen_nome_worker = QwenNomeBatchWorker(
            itens,
            model_path,
            int(self.settings.get("ocr", "qwen_max_new_tokens", 96)),
        )
        self._qwen_nome_worker.item_concluido.connect(self._on_qwen_nome_item)
        self._qwen_nome_worker.concluido.connect(self._on_qwen_nomes_concluido)
        self._qwen_nome_worker.erro.connect(self._on_qwen_erro)
        self._qwen_nome_worker.finished.connect(self._on_qwen_nomes_finalizado)
        self.btn_qwen_nomes.setEnabled(False)
        self.progresso.setVisible(True)
        self.progresso.setRange(0, len(itens))
        self.progresso.setValue(0)
        self.progresso.setFormat("Qwen corrigindo somente nomes incertos...")
        self._qwen_nome_worker.start()

    @Slot(dict)
    def _on_qwen_nome_item(self, resultado: dict) -> None:
        item = resultado.get("item") or {}
        campos = resultado.get("campos") or {}
        nome = tratar_valor(resultado.get("nome") or campos.get("nome_registrado") or "")
        data_registro = tratar_valor(campos.get("data_registro") or "")
        if nome or data_registro:
            if nome:
                self.repo.db.update(
                    """
                    UPDATE ocr_deteccao
                    SET status='superado', updated_at=datetime('now')
                    WHERE registro_id=? AND fonte='ocr_nome_rapido' AND ativo=1
                    """,
                    (item.get("registro_id"),),
                )
            execucao_id = self.repo.criar_execucao_ocr(
                imagem_id=item["imagem_id"],
                registro_id=item["registro_id"],
                motor=resultado.get("motor") or "qwen2-vl-2b-registro",
                texto_bruto=resultado.get("texto_bruto") or json.dumps(
                    campos, ensure_ascii=False
                ),
                tempo_ms=resultado.get("tempo_ms") or 0,
                sucesso=True,
                substituir_ativa=False,
            )
            self.repo.salvar_deteccoes_ocr(
                execucao_id=execucao_id,
                imagem_id=item["imagem_id"],
                registro_id=item["registro_id"],
                deteccoes=[
                    item
                    for item in (
                        {
                            "tipo": "nome_registrado",
                            "valor_original": nome,
                            "valor_tratado": nome,
                            "valor_normalizado": normalizar_busca(nome),
                            "confianca": 0.65,
                            "motor": resultado.get("motor") or "qwen2-vl-2b-registro",
                            "fonte": "qwen_nome_correcao",
                            "status": "precisa_revisao",
                            "contexto": "Correção Qwen de candidato OCR rápido; confirmar no revisor.",
                        }
                        if nome
                        else None,
                        {
                            "tipo": "data",
                            "valor_original": data_registro,
                            "valor_tratado": data_registro,
                            "valor_normalizado": normalizar_busca(data_registro),
                            "confianca": 0.45,
                            "motor": resultado.get("motor") or "qwen2-vl-2b-registro",
                            "fonte": "qwen_data_registro",
                            "status": "precisa_revisao",
                            "contexto": "Data do cabeçalho sugerida pelo Qwen; conferir no revisor.",
                        }
                        if data_registro
                        else None,
                    )
                    if item is not None
                ],
            )
        self.progresso.setValue(self.progresso.value() + 1)
        self._buscar()

    @Slot(int, int)
    def _on_qwen_nomes_concluido(self, processados: int, falhas: int) -> None:
        self._atualizar_estatisticas()
        self._buscar()
        self.statusBar().showMessage(
            f"Qwen terminou nomes incertos: {processados} processado(s), {falhas} falha(s)."
        )

    @Slot()
    def _on_qwen_nomes_finalizado(self) -> None:
        if self._qwen_nome_worker:
            self._qwen_nome_worker.deleteLater()
        self._qwen_nome_worker = None
        self.btn_qwen_nomes.setEnabled(True)
        self.progresso.setVisible(False)

    def _abrir_digitalizador(self) -> None:
        executable = Path(sys.executable)
        if getattr(sys, "frozen", False):
            target = executable.parent.parent / "DigitalizadorLivros" / "DigitalizadorLivros.exe"
            if target.is_file():
                ok, _pid = QProcess.startDetached(str(target), [], str(target.parent))
                if ok:
                    return
        project_root = Path(__file__).resolve().parents[2]
        app_path = project_root / "app.py"
        if app_path.is_file():
            ok, _pid = QProcess.startDetached(sys.executable, [str(app_path)], str(project_root))
            if ok:
                return
        QMessageBox.warning(
            self,
            "Digitalizador não encontrado",
            "Não foi possível localizar ou abrir o módulo de captura.",
        )

    def _importar_livro_organizado(self) -> None:
        if self._import_worker and self._import_worker.isRunning():
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            "Selecione a pasta raiz do livro (a que contém FRENTE e VERSO)",
            "D:\\",
        )
        if not folder:
            return
        self._audit_importacao = None
        self._import_worker = ImportacaoLivroWorker(self.db.path, Path(folder))
        self._import_worker.progresso.connect(self._on_progresso_importacao)
        self._import_worker.auditoria_concluida.connect(self._on_auditoria_importacao)
        self._import_worker.concluido.connect(self._on_importacao_concluida)
        self._import_worker.erro.connect(self._on_importacao_erro)
        self._import_worker.finished.connect(self._on_importacao_finalizada)
        self.progresso.setVisible(True)
        self.progresso.setRange(0, 0)
        self.progresso.setFormat("Auditando hashes, folhas e termos do A-16...")
        self._import_worker.start()

    @Slot(object)
    def _on_auditoria_importacao(self, audit: BookAudit) -> None:
        self._audit_importacao = audit
        if audit.nao_resolvidos:
            QMessageBox.warning(
                self,
                "Auditoria precisa de revisão",
                f"{len(audit.nao_resolvidos)} foto(s) de verso não puderam ser ligadas "
                "a uma folha com segurança. Nada foi gravado no acervo.\n\n"
                "A auditoria fica persistível pelo relatório; corrija essas faces no Revisor.",
            )
            return
        answer = QMessageBox.question(
            self,
            f"Importar Livro {audit.spec.codigo}",
            "Auditoria concluída sem alterar o banco:\n\n"
            f"• {len(audit.registros)} faces de registro localizadas\n"
            f"• {len(audit.indices)} páginas classificadas como índice\n"
            f"• {len(audit.faltantes)} faces ausentes\n"
            f"• {len(audit.duplicados)} duplicidades detectadas\n\n"
            "Os termos serão calculados pela folha auditada; uma ausência não deslocará "
            "as seguintes. Originais não serão modificados. Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        QTimer.singleShot(0, self._iniciar_importacao_auditada)

    def _iniciar_importacao_auditada(self) -> None:
        audit = self._audit_importacao
        if audit is None:
            return
        if self._import_worker and self._import_worker.isRunning():
            QTimer.singleShot(100, self._iniciar_importacao_auditada)
            return
        self._import_worker = ImportacaoLivroWorker(self.db.path, audit.root, audit)
        self._import_worker.progresso.connect(self._on_progresso_importacao)
        self._import_worker.concluido.connect(self._on_importacao_concluida)
        self._import_worker.erro.connect(self._on_importacao_erro)
        self._import_worker.finished.connect(self._on_importacao_finalizada)
        self.progresso.setVisible(True)
        self.progresso.setRange(0, len(audit.registros) + len(audit.indices))
        self._import_worker.start()

    @Slot(int, int, str)
    def _on_progresso_importacao(self, current: int, total: int, label: str) -> None:
        self.progresso.setMaximum(total)
        self.progresso.setValue(current)
        self.progresso.setFormat(f"Importando {current}/{total} — {label}")

    @Slot(dict)
    def _on_importacao_concluida(self, result: dict) -> None:
        QMessageBox.information(
            self,
            "Importação auditada concluída",
            f"{result.get('total_imagens', 0)} imagens associadas; "
            f"{result.get('novas_imagens', 0)} novas nesta execução; "
            f"{result.get('registros', 0)} registros pesquisáveis.\n"
            f"Faces ausentes: {len(result.get('faltantes') or [])}.\n\n"
            "OCR de nome/termo foi enfileirado somente para orientações aprovadas.",
        )
        self._carregar_filtros()
        self._atualizar_estatisticas()
        self._buscar()

    @Slot(str)
    def _on_importacao_erro(self, message: str) -> None:
        QMessageBox.warning(self, "Falha na importação", message)

    @Slot()
    def _on_importacao_finalizada(self) -> None:
        if self._import_worker:
            self._import_worker.deleteLater()
        self._import_worker = None
        self.progresso.setVisible(False)

    @Slot(int, int, str)
    def _on_progresso_indexacao(self, atual: int, total: int, nome: str) -> None:
        self.progresso.setMaximum(total)
        self.progresso.setValue(atual)
        self.progresso.setFormat(f"{atual}/{total} - {nome}")

    @Slot(str)
    def _on_erro_indexacao(self, mensagem: str) -> None:
        QMessageBox.warning(self, "Falha na indexação", mensagem)

    @Slot(int, int)
    def _on_indexacao_concluida(self, processadas: int, falhas: int) -> None:
        self.statusBar().showMessage(
            f"Indexação concluída: {processadas} processada(s), {falhas} falha(s)"
        )
        self._atualizar_estatisticas()
        self._buscar()

    @Slot()
    def _on_worker_finalizado(self) -> None:
        if self._worker:
            self._worker.deleteLater()
        self._worker = None
        self.btn_indexar.setEnabled(True)
        self.progresso.setVisible(False)

    def _atualizar_estatisticas(self) -> None:
        e = self.repo.estatisticas_consulta()
        self.lbl_estatisticas.setText(
            f"{e['acervos']} acervo(s) | {e['livros']} livro(s) | "
            f"{e['registros']} registro(s) | {e['nomes']} nome(s) detectado(s)"
        )

    def closeEvent(self, event) -> None:
        if self._processing_dialog and self._processing_dialog.running:
            QMessageBox.information(
                self,
                "Processamento em andamento",
                "Use 'Pausar após o item atual' no painel de Processamento e "
                "aguarde a confirmação antes de fechar.",
            )
            self._processing_dialog.show()
            self._processing_dialog.raise_()
            event.ignore()
            return
        if self._sync_worker and self._sync_worker.isRunning():
            QMessageBox.information(
                self,
                "Atualização em andamento",
                "Aguarde a atualização rápida dos registros antigos terminar antes de fechar.",
            )
            event.ignore()
            return
        if self._qwen_nome_worker and self._qwen_nome_worker.isRunning():
            QMessageBox.information(
                self,
                "Correção de nomes em andamento",
                "Aguarde a fila de nomes incertos do Qwen terminar.",
            )
            event.ignore()
            return
        if self._import_worker and self._import_worker.isRunning():
            self._import_worker.requestInterruption()
            if not self._import_worker.wait(30_000):
                QMessageBox.information(
                    self,
                    "Pausando importação",
                    "A etapa atual ainda está sendo salva. Tente fechar novamente em alguns "
                    "segundos; a retomada continuará do próximo arquivo.",
                )
                event.ignore()
                return
        if self._qwen_record_worker and self._qwen_record_worker.isRunning():
            QMessageBox.information(
                self,
                "Leitura Qwen em andamento",
                "Aguarde a leitura estruturada do registro terminar.",
            )
            event.ignore()
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Indexação em andamento",
                "Aguarde a indexação atual terminar antes de fechar.",
            )
            event.ignore()
            return
        if self._qwen_worker and self._qwen_worker.isRunning():
            QMessageBox.information(
                self,
                "Leitura Qwen em andamento",
                "Aguarde a leitura da área terminar antes de fechar.",
            )
            event.ignore()
            return
        if self._api_server is not None:
            self._api_server.stop()
            self._api_server = None
        self.db.close()
        event.accept()
