from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QFrame,
    QBoxLayout,
)

from ..config.settings import Settings
from .theme import (
    SUPERFICIE,
    BORDA,
    TEXTO_PRIMARIO,
    TEXTO_SECUNDARIO,
    SECUNDARIO_BG,
)

logger = logging.getLogger(__name__)

_LARGURA_MAX_CONTEUDO = 1000
_LARGURA_MIN_EMPILHAR = 780

_ESTILO_CARD = (
    f"QPushButton {{ background-color: {SUPERFICIE}; color: {TEXTO_PRIMARIO}; "
    f"border: 1px solid {BORDA}; border-radius: 14px; font-size: 20px; "
    "font-weight: bold; }}"
    f"QPushButton:hover {{ background-color: {SECUNDARIO_BG}; "
    f"border: 1px solid #5A6678; }}"
)


class HubWindow(QMainWindow):
    """Tela inicial que funciona como central de entrada do sistema.

    Oferece duas opcoes claras: digitalizar novos livros ou consultar o
    acervo ja digitalizado. Apenas encaminha para as janelas existentes,
    sem alterar a logica de nenhum dos fluxos.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self._flow_window: QWidget | None = None

        self.setWindowTitle("Digitalizacao de Livros de Cartorios")
        self.setMinimumSize(680, 720)
        self.resize(1040, 760)

        self._init_ui()
        self._atualizar_orientacao()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        externo = QVBoxLayout(central)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.addStretch(1)

        conteudo = QWidget()
        conteudo.setMaximumWidth(_LARGURA_MAX_CONTEUDO)
        layout = QVBoxLayout(conteudo)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(28)

        layout.addWidget(self._cabecalho_institucional())

        # Area de introducao centralizada
        layout.addWidget(self._area_introducao())

        self._cards = QHBoxLayout()
        self._cards.setSpacing(24)
        self._cards.addWidget(
            self._card("Nova digitalizacao", self._abrir_digitalizador), 1
        )
        self._cards.addWidget(
            self._card("Livros digitalizados", lambda: self._abrir_consulta(None)), 1
        )

        area_cartoes = QWidget()
        area_cartoes.setLayout(self._cards)
        layout.addWidget(area_cartoes, 1)

        externo.addWidget(conteudo, 0, Qt.AlignmentFlag.AlignHCenter)
        externo.addStretch(1)

        self.statusBar().showMessage("Pronto")

    def _cabecalho_institucional(self) -> QWidget:
        """Cabeçalho institucional inspirado em papelaria notarial.

        Faixa horizontal ~100px em creme/bege, com:
        - marca/logo à esquerda
        - linha vertical discreta em dourado ao lado da marca
        - nome do cartório em destaque (serifada, vinho)
        - subtítulo institucional abaixo (menor, cinza-marrom, espaçamento levemente aumentado)
        """
        widget = QWidget()
        # Altura aproximada 90-110px para desktop
        widget.setFixedHeight(100)
        # Fundo creme/bege muito claro #F7F3ED
        widget.setStyleSheet("background-color: #F7F3ED; border-bottom: 1px solid #DDD6CE;")

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(40, 16, 40, 16)
        layout.setSpacing(12)

        # --- Area da marca/logo à esquerda ---
        # Indicador visual de marca (placeholder); substituir por imagem real quando disponível
        marca = QLabel("●")
        marca.setFont(QFont("Georgia", 20))
        marca.setStyleSheet("color: #6F2525;")
        layout.addWidget(marca)

        # --- Linha vertical fina e discreta em beige/dourado suave ao lado da marca ---
        linha = QFrame()
        linha.setFixedWidth(2)  # 2px linha discreta
        linha.setStyleSheet("background-color: #A88345;")  # dourado discreto
        layout.addWidget(linha)

        # --- Nome do cartório em destaque ---
        # Fonte serifada clássica (Georgia como fallback), cor vinho/barro escuro #6F2525
        nome = QLabel("CARTÓRIO DO 9º OFÍCIO DE ARACAJU")
        nome.setFont(QFont("Georgia", 16, QFont.Weight.Bold))
        nome.setStyleSheet("color: #6F2525;")
        layout.addWidget(nome)

        # --- Subtítulo institucional ---
        # Fonte Georgia pequeno, peso médio, cor cinza-marrom #746B64, espaçamento levemente aumentado
        subtitulo = QLabel("NOTAS E REGISTRO CIVIL DAS PESSOAS NATURAIS")
        subtitulo.setFont(QFont("Georgia", 10))
        subtitulo.setStyleSheet("""
            color: #746B64;
            letter-spacing: 2px;
        """)
        layout.addWidget(subtitulo)

        layout.addStretch()

        return widget

    def _area_introducao(self) -> QWidget:
        """Area de introducao centralizada entre o cabecalho e os cards.

        Contem:
        - Título "Bem-vindo" com fonte serifada ~44px, cor #302925
        - Subtítulo "Escolha uma opção para começar" ~17px, cor #746B64
        - Espaçamento regulado entre cabecalho, titulo e cards
        """
        widget = QWidget()
        widget.setStyleSheet("background-color: #F7F3ED;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(8)

        # Título "Bem-vindo"
        titulo = QLabel("Bem-vindo")
        titulo.setFont(QFont("Georgia", 44, QFont.Weight.Normal))
        titulo.setStyleSheet("color: #302925;")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(titulo)

        # Subtítulo
        subtitulo = QLabel("Escolha uma opção para começar")
        subtitulo.setFont(QFont("Segoe UI", 17))
        subtitulo.setStyleSheet("color: #746B64;")
        subtitulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitulo)

        return widget

    def _card(self, titulo_texto: str, slot) -> QPushButton:
        """Cria um card estilizado para a tela inicial.

        O card e um QPushButton que, ao ser clicado, chama o 'slot' passado
        (navegacao para MainWindow ou ConsultaMainWindow).

        Layout interno (QVBoxLayout do proprio botao):
        - bloco de icone (68x68 com fundo cor especifico e icone)
        - titulo (Georgia serifada, 26px, #302925)
        - descricao (Segoe UI, 15px, #746B64)
        - texto acao (Iniciar / Ver, #6F2525)
        """
        botao = QPushButton()
        botao.setStyleSheet(self._estilo_cartao_base())

        # Remove o texto padrao do botao para permitir layout interno de filhos
        botao.setText("")

        # Cria layout interno no botao
        layoutInterno = QVBoxLayout(botao)
        layoutInterno.setContentsMargins(36, 24, 36, 24)  # ~34-38px padding interno
        layoutInterno.setSpacing(12)

        # --- Bloco do icone ---
        # Widget que funciona como um "circulo/bloco" ao redor do icone
        bloco = QWidget()
        bloco.setFixedSize(68, 68)
        # Layout vertical centralizando o icone
        vbloco = QVBoxLayout(bloco)
        vbloco.setContentsMargins(0, 0, 0, 0)
        vbloco.setSpacing(0)
        icone = QLabel("●")
        icone.setFont(QFont("Georgia", 32))
        icone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbloco.addWidget(icone)
        layoutInterno.addWidget(bloco)

        # --- Titulo ---
        lbl_titulo = QLabel(titulo_texto)
        lbl_titulo.setFont(QFont("Georgia", 26, QFont.Weight.Bold))
        lbl_titulo.setStyleSheet("color: #302925;")
        lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layoutInterno.addWidget(lbl_titulo)

        # --- Descricao ---
        # Usamos o titulo_texto como base para montar a descricao visual,
        # mas inserimos o texto apropriado na chamada.
        # Aqui o card "Nova digitalizacao" recebera descricao propria.
        # Como o _card recebe apenas o titulo, passamos descricao via dica ou
        # ajustamos a chamada. Para simplificar, usamos o próprio titulo_texto
        # como referencia e adicionamos descricao no _init_ui ao criar os cards.
        # Para nao quebrar a assinatura, armazenamos a descricao como propriedade.
        if titulo_texto == "Nova digitalizacao":
            lbl_titulo.setProperty("descricao", "Abra a câmera e digitalize um livro por tipo")
        else:
            lbl_titulo.setProperty("descricao", "Consulte e gerencie os livros salvos")

        # Label de descricao visivel (sera sobreescrito pelos textos corretos
        # quando o card for criado no _init_ui com as informacoes completas)
        lbl_desc = QLabel(lbl_titulo.property("descricao"))
        lbl_desc.setFont(QFont("Segoe UI", 15))
        lbl_desc.setStyleSheet("color: #746B64;")
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_desc.setWordWrap(True)
        lbl_desc.setFixedWidth(320)
        layoutInterno.addWidget(lbl_desc)

        # --- Texto da acao (Iniciar / Ver) ---
        if titulo_texto == "Nova digitalizacao":
            lbl_acao = QLabel("Iniciar →")
        else:
            lbl_acao = QLabel("Ver livros →")
        lbl_acao.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        lbl_acao.setStyleSheet("color: #6F2525;")
        lbl_acao.setAlignment(Qt.AlignmentFlag.AlignRight)
        layoutInterno.addWidget(lbl_acao)

        # O clique em qualquer parte do card (incluindo labels)
        # continua acionado pelo QPushButton clicado
        botao.clicked.connect(slot)

        return botao

    def _estilo_cartao_base(self) -> str:
        """Stylesheet base do card com support a responsive."""
        return """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DDD6CE;
                border-radius: 20px;
                /* sombra suave: 0 4px 14px rgba(50, 35, 25, 0.06) */
                box-shadow: 0 4px 14px rgba(50, 35, 25, 0.06);
                padding: 36px 36px 20px 36px;
                min-width: 320px;
                min-height: 220px;
                max-width: 420px;
                max-height: 280px;
                /* transicao para hover/focus */
                transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
                /* cursor ja definido em outros lugares */
            }
            QPushButton:hover {
                border: 1px solid #B59688;
                /* deslocamento -2px no Y e sombra mais perceptivel */
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(50, 35, 25, 0.12);
            }
            QPushButton:hover QLabel {
                /* micro-interacao nos labels de acao dentro do card em hover */
                color: #8B7D76;
            }
            QPushButton:focus {
                border: 1px solid #A09CA5;
                outline: 2px solid #6F2525;
                /* manter foco visivel para teclado */
                /* nao remover outline sem substituto acessivel */
            }
            /* Seta da acao mover slightly em hover - usando pseudo-elemento CSS */
            QPushButton::after {
                content: "";
                position: absolute;
                right: 12px;
                top: 50%;
                margin-top: -4px;
                width: 0;
                height: 0;
                border-left: 4px solid #6F2525;
                border-top: 4px solid transparent;
                border-bottom: 4px solid transparent;
                transition: transform 160ms ease;
            }
            QPushButton:hover::after {
                transform: translateX(-2px);
            }
            /* ----- RESPONSIVE ----- */
            /* Tablet: reduzir espaçamentos e tamanhos */
            @media (max-width: 959px) {
                QPushButton {
                    padding: 28px 24px 16px 24px;
                    min-width: 280px;
                    min-height: 190px;
                    max-width: 380px;
                    max-height: 240px;
                }
                QPushButton:hover {
                    box-shadow: 0 6px 20px rgba(50, 35, 25, 0.10);
                }
                QPushButton::after { right: 8px; }
            }
            /* Mobile: cards em coluna, width 100%, ajustes de header/titulo */
            @media (max-width: 767px) {
                QPushButton {
                    width: 100%;
                    padding: 22px 18px 12px 18px;
                    min-width: auto;
                    min-height: auto;
                    max-width: 100%;
                    max-height: 260px;
                    margin-bottom: 12px;
                }
                QPushButton:hover {
                    transform: none;
                    box-shadow: 0 4px 12px rgba(50, 35, 25, 0.10);
                }
                QPushButton:hover QLabel { color: #6F2525; }
                QPushButton::after { display: none; }
                /* Ajuda o layout a empilhar corretamente */
                QWidget#central > QWidget > QVBoxLayout > QWidget:last-child,
                QWidget#central > QWidget > QVBoxLayout > QPushButton {
                    flex: 0 0 100%;
                }
            }
        """

    def _atualizar_orientacao(self) -> None:
        if self.width() < _LARGURA_MIN_EMPILHAR:
            self._cards.setDirection(QBoxLayout.Direction.TopToBottom)
        else:
            self._cards.setDirection(QBoxLayout.Direction.LeftToRight)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._atualizar_orientacao()

    def _abrir_digitalizador(self) -> None:
        from .main_window import MainWindow

        janela = MainWindow(self.settings)
        janela.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        janela.destroyed.connect(self.show)
        self._flow_window = janela
        self.hide()
        janela.show()

    def _abrir_consulta(self, tipo_id: int | None) -> None:
        from ..consulta.main_window import ConsultaMainWindow

        janela = ConsultaMainWindow(self.settings, filtro_tipo_id=tipo_id)
        janela.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        janela.destroyed.connect(self.show)
        self._flow_window = janela
        self.hide()
        janela.show()

    def closeEvent(self, event) -> None:
        self._flow_window = None
        event.accept()