from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

import cv2
import numpy as np

from ..imaging.document import detectar_quadrilatero_pagina


class CaptureState(Enum):
    """Estados da captura. Os valores sao as mensagens exibidas ao operador."""

    SEM_FOLHA = "Posicione a pagina"
    MAO_PRESENTE = "Retire a mao"
    NAO_ENQUADRADA = "Afaste ou centralize a pagina"
    AGUARDANDO_FOCO = "Aguardando foco"
    AGUARDANDO_ESTABILIDADE = "Aguarde estabilizar"
    PAGINA_PRONTAA = "Pagina pronta"
    CAPTURADA = "Capturada"
    TROQUE_PAGINA = "Troque a pagina"
    CAPTURA_MANUAL = "Pronta para captura manual"


@dataclass
class FrameAnalysis:
    movimento: float
    foco: float
    mao_presente: bool
    pagina_presente: bool
    status: str
    contagem: float | None = None
    capturar: bool = False
    enquadrada: bool = True
    pagina_contorno: np.ndarray | None = None
    estado: CaptureState | None = None


def _reduzir(frame: np.ndarray, largura: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= largura:
        return frame
    escala = largura / w
    return cv2.resize(
        frame,
        (largura, max(1, int(h * escala))),
        interpolation=cv2.INTER_AREA,
    )


def pontuacao_mao(frame: np.ndarray) -> float:
    """Estima a maior oclusao de pele; e um filtro conservador e rapido."""
    pequena = _reduzir(frame, 480)
    h, w = pequena.shape[:2]
    ycrcb = cv2.cvtColor(pequena, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(pequena, cv2.COLOR_BGR2HSV)
    _, cr, cb = cv2.split(ycrcb)
    _, saturacao, valor = cv2.split(hsv)
    azul, verde, vermelho = cv2.split(pequena)
    vermelho_i = vermelho.astype(np.int16)
    verde_i = verde.astype(np.int16)
    mascara = (
        (cr >= 132)
        & (cr <= 185)
        & (cb >= 75)
        & (cb <= 140)
        & (saturacao >= 35)
        & (valor >= 45)
        & (vermelho_i > verde_i + 5)
    ).astype(np.uint8) * 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    total, _, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if total <= 1:
        return 0.0
    maior = int(stats[1:, cv2.CC_STAT_AREA].max())
    return maior / float(max(1, h * w))


class AutoCaptureController:
    """Decide quando capturar sem repetir a mesma pagina.

    A pagina precisa ficar nitida, sem grande oclusao e praticamente parada
    durante ``tempo_estavel``. Depois da foto, uma mudanca real de cena e
    obrigatoria antes que outra captura seja liberada.
    """

    def __init__(
        self,
        tempo_estavel: float = 1.2,
        movimento_maximo: float = 2.4,
        mudanca_pagina: float = 10.0,
        foco_minimo: float = 55.0,
        mao_maxima: float = 0.20,
        tempo_troca: float = 0.6,
    ) -> None:
        self.tempo_estavel = tempo_estavel
        self.movimento_maximo = movimento_maximo
        self.mudanca_pagina = mudanca_pagina
        self.foco_minimo = foco_minimo
        self.mao_maxima = mao_maxima
        # Tempo minimo com a pagina NOVA presente e estavel para destravar o
        # cooldown apos uma captura (M2-T02). Impede recaptura da mesma folha.
        self.tempo_troca = tempo_troca
        self._anterior: np.ndarray | None = None
        self._capturada: np.ndarray | None = None
        self._estavel_desde: float | None = None
        self._troca_desde: float | None = None
        self._bloqueada = False

    @staticmethod
    def _cinza(frame: np.ndarray) -> np.ndarray:
        pequena = _reduzir(frame, 480)
        return cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY)

    def analisar(self, frame: np.ndarray, agora: float | None = None) -> FrameAnalysis:
        agora = time.monotonic() if agora is None else agora
        cinza = self._cinza(frame)
        movimento = (
            float(cv2.absdiff(cinza, self._anterior).mean())
            if self._anterior is not None and self._anterior.shape == cinza.shape
            else 999.0
        )
        self._anterior = cinza

        foco = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
        centro = cinza[
            int(cinza.shape[0] * 0.05):int(cinza.shape[0] * 0.95),
            int(cinza.shape[1] * 0.08):int(cinza.shape[1] * 0.92),
        ]
        pagina_presente = bool(centro.size and centro.mean() >= 60 and centro.std() >= 18)
        contorno = detectar_quadrilatero_pagina(frame, largura=480)
        enquadrada = True
        if contorno is not None:
            pagina_presente = True
            min_x, max_x = float(contorno[:, 0].min()), float(contorno[:, 0].max())
            min_y, max_y = float(contorno[:, 1].min()), float(contorno[:, 1].max())
            margem = 0.018
            enquadrada = (
                min_x > margem and min_y > margem
                and max_x < 1.0 - margem and max_y < 1.0 - margem
            )
        mao_score = pontuacao_mao(frame)
        mao_presente = mao_score >= self.mao_maxima

        def _resultado(
            estado: CaptureState,
            *,
            movimento_f: float = movimento,
            foco_f: float = foco,
            mao: bool = mao_presente,
            pagina: bool = pagina_presente,
            contagem: float | None = None,
            enquadrada_f: bool = enquadrada,
            contorno_f: np.ndarray | None = contorno,
            capturar: bool = False,
        ) -> FrameAnalysis:
            return FrameAnalysis(
                movimento_f, foco_f, mao, pagina, estado.value,
                contagem, capturar, enquadrada_f, contorno_f, estado,
            )

        if self._bloqueada:
            if mao_presente:
                self._troca_desde = None
                return _resultado(CaptureState.MAO_PRESENTE)
            if not pagina_presente:
                self._troca_desde = None
                return _resultado(CaptureState.TROQUE_PAGINA)
            mudanca = (
                float(cv2.absdiff(cinza, self._capturada).mean())
                if self._capturada is not None and self._capturada.shape == cinza.shape
                else 999.0
            )
            # A troca so destrava com uma pagina NOVA, presente e estavel por
            # tempo_troca. A mesma folha voltando ou um frame sem pagina nao
            # podem liberar recaptura (achado D-011, corrigido no M2-T02).
            if mudanca >= self.mudanca_pagina and movimento <= self.movimento_maximo:
                if self._troca_desde is None:
                    self._troca_desde = agora
                    return _resultado(CaptureState.TROQUE_PAGINA)
                if agora - self._troca_desde < self.tempo_troca:
                    return _resultado(CaptureState.TROQUE_PAGINA)
                self._bloqueada = False
                self._troca_desde = None
                self._capturada = None
                self._estavel_desde = None
            else:
                self._troca_desde = None
                return _resultado(CaptureState.TROQUE_PAGINA)

        if not pagina_presente:
            self._estavel_desde = None
            return _resultado(CaptureState.SEM_FOLHA)
        if mao_presente:
            self._estavel_desde = None
            return _resultado(CaptureState.MAO_PRESENTE)
        if not enquadrada:
            self._estavel_desde = None
            return _resultado(CaptureState.NAO_ENQUADRADA)
        if foco < self.foco_minimo:
            self._estavel_desde = None
            return _resultado(CaptureState.AGUARDANDO_FOCO)
        if movimento > self.movimento_maximo:
            self._estavel_desde = None
            return _resultado(CaptureState.AGUARDANDO_ESTABILIDADE)

        if self._estavel_desde is None:
            self._estavel_desde = agora
        restante = max(0.0, self.tempo_estavel - (agora - self._estavel_desde))
        if restante > 0:
            return _resultado(CaptureState.PAGINA_PRONTAA, contagem=restante)

        self._bloqueada = True
        self._capturada = cinza.copy()
        self._estavel_desde = None
        self._troca_desde = None
        return _resultado(CaptureState.CAPTURADA, contagem=0.0, capturar=True)

    def marcar_capturada(self, frame: np.ndarray) -> None:
        self._capturada = self._cinza(frame)
        self._anterior = self._capturada.copy()
        self._bloqueada = True
        self._estavel_desde = None
        self._troca_desde = None

    def reset(self) -> None:
        self._anterior = None
        self._capturada = None
        self._estavel_desde = None
        self._troca_desde = None
        self._bloqueada = False
