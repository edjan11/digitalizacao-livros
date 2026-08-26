"""Tema Dark Mode Enterprise de ALTO CONTRASTE (WCAG AA/AAA).

Apenas estilo. Cores padronizadas para legibilidade perfeita: textos claros
sobre fundos escuros, bordas nitidas e acoes com contraste acessivel.
"""

from __future__ import annotations

# ---- Camadas / superficies ------------------------------------------------
BG_GERAL = "#12151C"          # fundo geral quase preto
SUPERFICIE = "#1E232D"        # cards / containers (cinza azulado escuro)
BORDA = "#343B48"             # divisores e bordas de painel
VIEWPORT_BG = "#0F172A"       # area central de exibicao
VIEWPORT_BORDA = "#334155"    # borda de destaque da viewport

# ---- Textos ---------------------------------------------------------------
TEXTO_PRIMARIO = "#FFFFFF"    # titulos / texto principal / botoes
TEXTO_NEON = "#F0F4F8"        # cinza claro neon (primario alternativo)
TEXTO_SECUNDARIO = "#A0AEC0"  # rotulos / subtitulos
TEXTO_DESATIVADO = "#64748B"  # desabilitado (ainda visivel)

# ---- Acoes ----------------------------------------------------------------
VERDE_ESMERALDA = "#10B981"   # acao primaria
VERDE_ESMERALDA_HOVER = "#059669"
SECUNDARIO_BG = "#2D3748"     # acao secundaria
SECUNDARIO_BORDA = "#4A5568"
ALERTA_BG = "#DC2626"         # revisar pendentes / atencao

# ---- Status (glyphs/texto sobre fundos escuros, alto contraste) ----------
STATUS_OK = "#34D399"         # verde esmeralda claro
STATUS_ATENCAO = "#FBBF24"    # ocre/ambar
STATUS_ERRO = "#F87171"       # vermelho claro (legivel no escuro)

# Glyphs (icones vetoriais simples, Segoe UI)
G_IR_ESQ = "↺"
G_IR_DIR = "↻"
G_ESPELHAR = "⇄"
G_CORTAR = "✂"
G_OTIMIZAR = "✦"
G_IMAGEM = "◻"
G_FOCO = "◎"
G_ENQUAD = "▢"
G_DUP = "⧉"
G_OCR = "T"
G_SCANNER = "▣"

QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG_GERAL};
    color: {TEXTO_NEON};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11px;
}}
QLabel {{ color: {TEXTO_NEON}; }}
QFrame#panel {{
    background-color: {SUPERFICIE};
    border: 1px solid {BORDA};
    border-radius: 6px;
}}
QGroupBox {{
    background-color: {SUPERFICIE};
    border: 1px solid {BORDA};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    color: {TEXTO_SECUNDARIO};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXTO_SECUNDARIO};
    font-weight: bold;
}}
QPushButton {{
    background-color: {SECUNDARIO_BG};
    color: {TEXTO_PRIMARIO};
    border: 1px solid {SECUNDARIO_BORDA};
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: #3B4757; border: 1px solid #5A6678; }}
QPushButton:pressed {{ background-color: #1A2231; }}
QPushButton:disabled {{
    background-color: {SUPERFICIE};
    color: {TEXTO_DESATIVADO};
    border: 1px solid {BORDA};
}}
QGraphicsView, QScrollArea {{
    background-color: {VIEWPORT_BG};
    border: 1px solid {VIEWPORT_BORDA};
}}
QProgressBar {{
    background-color: {SUPERFICIE};
    border: 1px solid {BORDA};
    border-radius: 4px;
    text-align: center;
    color: {TEXTO_PRIMARIO};
}}
QProgressBar::chunk {{ background-color: {VERDE_ESMERALDA}; border-radius: 3px; }}
QPlainTextEdit, QTextEdit, QLineEdit {{
    background-color: {VIEWPORT_BG};
    color: {TEXTO_NEON};
    border: 1px solid {BORDA};
    border-radius: 4px;
}}
QPlainTextEdit:disabled, QLineEdit:disabled {{
    color: {TEXTO_DESATIVADO};
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {VIEWPORT_BG};
    color: {TEXTO_PRIMARIO};
    border: 1px solid {BORDA};
    border-radius: 4px;
    padding: 3px 5px;
}}
QComboBox QAbstractItemView {{
    background-color: {SUPERFICIE};
    color: {TEXTO_PRIMARIO};
    selection-background-color: {SECUNDARIO_BG};
}}
QMenu {{
    background-color: {SUPERFICIE};
    color: {TEXTO_PRIMARIO};
    border: 1px solid {BORDA};
}}
QMenu::item:selected {{ background-color: {SECUNDARIO_BG}; }}
QMenu::item:disabled {{ color: {TEXTO_DESATIVADO}; }}
QTabWidget::pane {{ border: 1px solid {BORDA}; }}
QTableWidget, QTreeWidget, QListWidget {{
    background-color: {VIEWPORT_BG};
    color: {TEXTO_NEON};
    gridline-color: {BORDA};
}}
QHeaderView::section {{
    background-color: {SUPERFICIE};
    color: {TEXTO_SECUNDARIO};
    border: 1px solid {BORDA};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {BG_GERAL};
    width: 10px; height: 10px;
}}
QScrollBar::handle {{
    background: {BORDA};
    border-radius: 5px;
}}
QScrollBar::handle:hover {{ background: {SECUNDARIO_BORDA}; }}
QCheckBox {{ color: {TEXTO_NEON}; spacing: 6px; }}
QRadioButton {{ color: {TEXTO_NEON}; spacing: 6px; }}
QToolTip {{
    background-color: {SUPERFICIE};
    color: {TEXTO_PRIMARIO};
    border: 1px solid {BORDA};
}}
"""

# Estilos inline das acoes de estado (cores solidas acessiveis)
BTN_PRIMARIO = (
    f"QPushButton {{ background-color: {VERDE_ESMERALDA}; color: {TEXTO_PRIMARIO}; "
    "border: none; border-radius: 4px; padding: 6px 14px; font-weight: bold; } "
    f"QPushButton:hover {{ background-color: {VERDE_ESMERALDA_HOVER}; }}"
)
BTN_SECUNDARIO = (
    f"QPushButton, QToolButton {{ background-color: {SECUNDARIO_BG}; color: {TEXTO_PRIMARIO}; "
    f"border: 1px solid {SECUNDARIO_BORDA}; border-radius: 4px; padding: 6px 14px; font-weight: bold; }} "
    f"QPushButton:hover, QToolButton:hover {{ background-color: #3B4757; border: 1px solid #5A6678; }} "
    "QToolButton::menu-indicator { subcontrol-position: right center; }"
)
BTN_ALERTA = (
    f"QPushButton {{ background-color: {ALERTA_BG}; color: {TEXTO_PRIMARIO}; "
    "border: none; border-radius: 4px; padding: 6px 14px; font-weight: bold; } "
    f"QPushButton:hover {{ background-color: #B91C1C; }}"
)

STATUS_CORES = {
    "ok": STATUS_OK, "confirmado": STATUS_OK, "provavel": STATUS_OK,
    "inferido_sequencia": STATUS_OK, "aviso": STATUS_ATENCAO, "duvidoso": STATUS_ATENCAO,
    "revisar": STATUS_ATENCAO, "precisa_revisao": STATUS_ATENCAO, "erro_grave": STATUS_ERRO,
}
STATUS_GLYPH = {
    "ok": "✓", "confirmado": "✓", "provavel": "✓",
    "inferido_sequencia": "✓", "aviso": "⚠", "duvidoso": "⚠",
    "revisar": "⚠", "precisa_revisao": "⚠", "erro_grave": "✕",
}


def status_visual(status: str) -> tuple[str, str]:
    """Devolve (glyph, cor) para um status de indicador."""
    return STATUS_GLYPH.get(status, "•"), STATUS_CORES.get(status, TEXTO_SECUNDARIO)
