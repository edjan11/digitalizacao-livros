from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..services.name_processing import NameBatchRunner
from .theme import VERDE_ESMERALDA, VERDE_ESMERALDA_HOVER, TEXTO_PRIMARIO, STATUS_OK, STATUS_ERRO, STATUS_ATENCAO, SUPERFICIE, BORDA, TEXTO_NEON

class NameProcessingWorker(QThread):
    progresso = Signal(dict, str)
    concluido = Signal(dict)
    erro = Signal(str)

    def __init__(self, *, db_path: Path, settings, lote_id: int, max_workers: int = 4):
        super().__init__()
        self.db_path = Path(db_path)
        self.settings = settings
        self.lote_id = int(lote_id)
        self.max_workers = int(max_workers)

    def run(self) -> None:
        try:
            runner = NameBatchRunner(
                db_path=self.db_path,
                settings=self.settings,
                lote_id=self.lote_id,
                max_workers=self.max_workers,
                should_stop=self.isInterruptionRequested,
                on_progress=lambda resumo, label: self.progresso.emit(resumo, label),
            )
            self.concluido.emit(runner.run())
        except Exception as exc:
            self.erro.emit(str(exc))


class ProcessingDialog(QDialog):
    def __init__(self, *, repo, settings, abrir_registro=None, parent=None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.settings = settings
        self.abrir_registro = abrir_registro
        self.worker: NameProcessingWorker | None = None
        self.lote: dict | None = None
        self._ultimo_label = "Fila preparada"
        self._auto_resume_checked = False
        self.setWindowTitle("Processamento dos nomes — histórico persistente")
        self.setMinimumSize(1180, 700)
        self.resize(1450, 840)
        self._init_ui()
        self._carregar_livros()
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.atualizar)
        self.timer.start()

    @property
    def running(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def _init_ui(self) -> None:
        raiz = QVBoxLayout(self)
        topo = QHBoxLayout()
        topo.addWidget(QLabel("Livro:"))
        self.combo_livro = QComboBox()
        self.combo_livro.currentIndexChanged.connect(self._trocar_livro)
        topo.addWidget(self.combo_livro, 1)
        topo.addWidget(QLabel("Etapa:"))
        self.combo_etapa = QComboBox()
        self.combo_etapa.addItem("Todas", None)
        self.combo_etapa.addItem("OCR rápido", "ocr_nome_rapido")
        self.combo_etapa.addItem("Qwen — nome", "qwen_nome")
        self.combo_etapa.currentIndexChanged.connect(self.atualizar)
        topo.addWidget(self.combo_etapa)
        topo.addWidget(QLabel("Status:"))
        self.combo_status = QComboBox()
        self.combo_status.addItem("Todos", None)
        for status in (
            "pendente", "processando", "sugestao", "revisar", "confirmado",
            "sem_resultado", "falhou", "pausado",
        ):
            self.combo_status.addItem(status.replace("_", " ").title(), status)
        self.combo_status.currentIndexChanged.connect(self.atualizar)
        topo.addWidget(self.combo_status)
        raiz.addLayout(topo)

        cards = QGridLayout()
        self.cards: dict[str, QLabel] = {}
        nomes = (
            ("total", "TOTAL"),
            ("rapido", "OCR RÁPIDO"),
            ("sugestoes", "SUGESTÕES"),
            ("qwen", "AGUARDANDO QWEN"),
            ("revisar", "AGUARDANDO OPERADOR"),
            ("confirmados", "CONFIRMADOS"),
            ("sem_resultado", "SEM RESULTADO"),
            ("falhas", "FALHAS"),
        )
        for coluna, (chave, titulo) in enumerate(nomes):
            label = QLabel(f"{titulo}\n0")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                "background:#f5f7fa; border:1px solid #cfd8dc; border-radius:5px; "
                "padding:8px; font-weight:bold; color:#263238;"
            )
            cards.addWidget(label, 0, coluna)
            self.cards[chave] = label
        raiz.addLayout(cards)

        self.progresso = QProgressBar()
        self.progresso.setRange(0, 1000)
        self.progresso.setFormat("Fila ainda não preparada")
        raiz.addWidget(self.progresso)
        self.lbl_detalhes = QLabel("O histórico permanece salvo mesmo após fechar o aplicativo.")
        self.lbl_detalhes.setStyleSheet("color:#455a64;")
        raiz.addWidget(self.lbl_detalhes)

        acoes = QHBoxLayout()
        self.btn_iniciar = QPushButton("Iniciar / Retomar")
        self.btn_iniciar.setStyleSheet(
            f"QPushButton {{ background-color: {VERDE_ESMERALDA}; color: {TEXTO_PRIMARIO}; "
            f"font-weight: bold; padding: 7px 16px; border: none; }} "
            f"QPushButton:hover {{ background-color: {VERDE_ESMERALDA_HOVER}; }}"
        )
        self.btn_iniciar.clicked.connect(self.iniciar)
        acoes.addWidget(self.btn_iniciar)
        self.btn_pausar = QPushButton("Pausar após o item atual")
        self.btn_pausar.clicked.connect(self.pausar)
        self.btn_pausar.setEnabled(False)
        acoes.addWidget(self.btn_pausar)
        falhas = QPushButton("Reprocessar falhas")
        falhas.clicked.connect(self.reprocessar_falhas)
        acoes.addWidget(falhas)
        abrir = QPushButton("Abrir no Revisor")
        abrir.clicked.connect(self._abrir_selecionado)
        acoes.addWidget(abrir)
        exportar = QPushButton("Exportar histórico")
        exportar.clicked.connect(self._exportar)
        acoes.addWidget(exportar)
        acoes.addStretch()
        raiz.addLayout(acoes)

        colunas = [
            "Termo", "Folha", "Face", "Foto", "Etapa", "Motor",
            "Resultado", "Confiança", "Tempo", "Status", "Erro",
        ]
        self.tabela = QTableWidget(0, len(colunas))
        self.tabela.setHorizontalHeaderLabels(colunas)
        self.tabela.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tabela.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tabela.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tabela.verticalHeader().setVisible(False)
        self.tabela.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.tabela.horizontalHeader().setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)
        raiz.addWidget(self.tabela, 1)

    def _carregar_livros(self) -> None:
        self.combo_livro.blockSignals(True)
        self.combo_livro.clear()
        for livro in self.repo.listar_acervo_livros():
            if int(livro.get("total_registros") or 0) <= 0:
                continue
            self.combo_livro.addItem(
                f"{livro.get('codigo') or '?'} — {livro.get('total_registros') or 0} registros",
                int(livro["id"]),
            )
        self.combo_livro.blockSignals(False)
        self._trocar_livro()

    @Slot()
    def _trocar_livro(self) -> None:
        livro_id = self.combo_livro.currentData()
        if livro_id is None:
            self.lote = None
            return
        self.lote = self.repo.criar_ou_sincronizar_lote_nomes(int(livro_id))
        self.atualizar()

    def auto_resume(self) -> None:
        if self._auto_resume_checked:
            return
        self._auto_resume_checked = True
        if self.lote and self.lote.get("status") == "processando":
            self.iniciar()

    @Slot()
    def iniciar(self) -> None:
        if self.running or not self.lote:
            return
        self.repo.marcar_lote_status(int(self.lote["id"]), "processando")
        self.worker = NameProcessingWorker(
            db_path=self.repo.db.path,
            settings=self.settings,
            lote_id=int(self.lote["id"]),
            max_workers=4,
        )
        self.worker.progresso.connect(self._on_progresso)
        self.worker.concluido.connect(self._on_concluido)
        self.worker.erro.connect(self._on_erro)
        self.worker.finished.connect(self._on_finalizado)
        self.btn_iniciar.setEnabled(False)
        self.btn_pausar.setEnabled(True)
        self.worker.start()

    @Slot()
    def pausar(self) -> None:
        if not self.running:
            return
        self.worker.requestInterruption()
        self.btn_pausar.setEnabled(False)
        self.lbl_detalhes.setText(
            "Pausa solicitada; o item atual será preservado antes de parar."
        )

    @Slot()
    def reprocessar_falhas(self) -> None:
        if not self.lote or self.running:
            return
        total = self.repo.reprocessar_falhas_lote(int(self.lote["id"]))
        QMessageBox.information(self, "Fila", f"{total} falha(s) voltaram para a fila.")
        self.atualizar()

    @Slot(dict, str)
    def _on_progresso(self, _resumo: dict, label: str) -> None:
        self._ultimo_label = label
        self.atualizar()

    @Slot(dict)
    def _on_concluido(self, resumo: dict) -> None:
        self._ultimo_label = (
            "Fila pausada com segurança." if resumo.get("status") == "pausado" else "Fila concluída."
        )
        self.atualizar()

    @Slot(str)
    def _on_erro(self, mensagem: str) -> None:
        QMessageBox.warning(self, "Processamento", mensagem)

    @Slot()
    def _on_finalizado(self) -> None:
        if self.worker:
            self.worker.deleteLater()
        self.worker = None
        self.btn_iniciar.setEnabled(True)
        self.btn_pausar.setEnabled(False)
        self.atualizar()

    @staticmethod
    def _formatar_tempo(segundos: float) -> str:
        if segundos <= 0:
            return "calculando"
        return str(timedelta(seconds=int(segundos)))

    @Slot()
    def atualizar(self) -> None:
        if not self.lote:
            return
        resumo = self.repo.resumo_processamento(int(self.lote["id"]))
        self.lote = resumo
        c = resumo.get("contagens") or {}
        total = int(resumo.get("total_registros") or 0)
        pendente = int(c.get("ocr_nome_rapido:pendente", 0))
        processando = int(c.get("ocr_nome_rapido:processando", 0))
        rapido = max(0, total - pendente - processando)
        sugestoes = int(c.get("ocr_nome_rapido:sugestao", 0))
        qwen = int(c.get("qwen_nome:pendente", 0)) + int(c.get("qwen_nome:processando", 0))
        revisar = int(c.get("ocr_nome_rapido:revisar", 0)) + int(c.get("qwen_nome:revisar", 0))
        sem_resultado = int(c.get("ocr_nome_rapido:sem_resultado", 0)) + int(c.get("qwen_nome:sem_resultado", 0))
        falhas = int(c.get("ocr_nome_rapido:falhou", 0)) + int(c.get("qwen_nome:falhou", 0))
        valores = {
            "total": total,
            "rapido": rapido,
            "sugestoes": sugestoes,
            "qwen": qwen,
            "revisar": revisar,
            "confirmados": int(resumo.get("confirmados") or 0),
            "sem_resultado": sem_resultado,
            "falhas": falhas,
        }
        titulos = {
            "total": "TOTAL", "rapido": "OCR RÁPIDO", "sugestoes": "SUGESTÕES",
            "qwen": "AGUARDANDO QWEN", "revisar": "AGUARDANDO OPERADOR",
            "confirmados": "CONFIRMADOS", "sem_resultado": "SEM RESULTADO", "falhas": "FALHAS",
        }
        for chave, valor in valores.items():
            self.cards[chave].setText(f"{titulos[chave]}\n{valor}")
        fracao = rapido / max(1, total)
        self.progresso.setValue(int(fracao * 1000))
        self.progresso.setFormat(
            f"OCR rápido: {rapido}/{total} ({fracao * 100:.1f}%) — lote {resumo.get('status') or 'pendente'}"
        )
        medias = resumo.get("medias_ms") or {}
        restante_segundos = 0.0
        if float(medias.get("ocr_nome_rapido") or 0) > 0:
            restante_segundos += (
                (pendente + processando)
                * float(medias["ocr_nome_rapido"])
                / 1000
                / 4
            )
        if float(medias.get("qwen_nome") or 0) > 0:
            restante_segundos += qwen * float(medias["qwen_nome"]) / 1000
        eta = self._formatar_tempo(restante_segundos)
        self.lbl_detalhes.setText(
            f"{self._ultimo_label} — estimativa restante: {eta}"
        )
        # O lote pode continuar em segundo plano com esta janela escondida.
        # Nesse caso, atualizar milhares de células a cada dois segundos só
        # roubaria CPU do OCR; os contadores continuam sendo atualizados.
        if self.isVisible():
            self._preencher_tabela()

    def _itens_filtrados(self) -> list[dict]:
        if not self.lote:
            return []
        status = self.combo_status.currentData()
        return self.repo.listar_itens_processamento(
            int(self.lote["id"]),
            etapa=self.combo_etapa.currentData(),
            statuses=(str(status),) if status else None,
            limite=5000,
        )

    def _preencher_tabela(self) -> None:
        itens = self._itens_filtrados()
        self.tabela.setRowCount(len(itens))
        cores = {
            "confirmado": (SUPERFICIE, STATUS_OK),
            "falhou": (SUPERFICIE, STATUS_ERRO),
            "revisar": (SUPERFICIE, STATUS_ATENCAO),
            "processando": (SUPERFICIE, TEXTO_NEON),
            "pendente": (SUPERFICIE, TEXTO_NEON),
            "sugestao": (SUPERFICIE, TEXTO_NEON),
        }
        for linha, item in enumerate(itens):
            valores = [
                item.get("termo") or "?",
                item.get("folha") or "?",
                item.get("face") or "?",
                Path(item.get("caminho_original") or "").name,
                item.get("etapa") or "?",
                item.get("motor") or "—",
                item.get("resultado") or "—",
                f"{float(item.get('confianca') or 0) * 100:.0f}%",
                f"{float(item.get('tempo_ms') or 0) / 1000:.1f}s",
                item.get("status") or "?",
                item.get("erro") or "",
            ]
            fundo, frente = cores.get(str(item.get("status")), ("#ffffff", "#37474f"))
            for coluna, valor in enumerate(valores):
                celula = QTableWidgetItem(str(valor))
                celula.setBackground(QColor(fundo))
                celula.setForeground(QColor(frente))
                if coluna == 0:
                    celula.setData(Qt.ItemDataRole.UserRole, int(item["registro_id"]))
                self.tabela.setItem(linha, coluna, celula)
        self.tabela.resizeColumnsToContents()
        self.tabela.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.tabela.horizontalHeader().setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)

    @Slot()
    def _abrir_selecionado(self) -> None:
        linha = self.tabela.currentRow()
        celula = self.tabela.item(linha, 0) if linha >= 0 else None
        registro_id = celula.data(Qt.ItemDataRole.UserRole) if celula else None
        if registro_id is not None and self.abrir_registro:
            self.abrir_registro(int(registro_id))

    @Slot()
    def _exportar(self) -> None:
        itens = self._itens_filtrados()
        if not itens:
            QMessageBox.information(self, "Histórico", "Não há itens neste filtro.")
            return
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Exportar histórico", "historico_processamento.csv", "CSV (*.csv)"
        )
        if not caminho:
            return
        campos = [
            "registro_id", "termo", "folha", "face", "caminho_original", "etapa",
            "status", "motor", "resultado", "confianca", "tempo_ms", "tentativas",
            "erro", "bbox_json", "iniciado_em", "concluido_em",
        ]
        with open(caminho, "w", newline="", encoding="utf-8-sig") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=campos)
            writer.writeheader()
            for item in itens:
                writer.writerow({campo: item.get(campo) for campo in campos})

    def closeEvent(self, event) -> None:
        if self.running:
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.atualizar()
