from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.auto_capture import AutoCaptureController, pontuacao_mao
from src.imaging.quality import avaliar_qualidade, detectar_dobra_grande


def _pagina(seed: int = 1) -> np.ndarray:
    image = np.full((720, 960, 3), 225, np.uint8)
    cv2.rectangle(image, (90, 45), (870, 675), (250, 250, 250), -1)
    cv2.rectangle(image, (90, 45), (870, 675), (25, 25, 25), 5)
    for y in range(100, 640, 30):
        cv2.line(image, (125, y), (835, y), (65, 65, 65), 2)
    cv2.putText(
        image, f"Numero {6800 + seed}", (135, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2,
    )
    return image


def test_captura_so_depois_de_estavel_e_nao_repete_a_pagina():
    pagina = _pagina(1)
    controller = AutoCaptureController(tempo_estavel=1.0)

    assert not controller.analisar(pagina, agora=0.0).capturar
    assert not controller.analisar(pagina, agora=0.1).capturar
    assert controller.analisar(pagina, agora=1.2).capturar

    repetida = controller.analisar(pagina, agora=2.5)
    assert repetida.status == "Troque a pagina"
    assert not repetida.capturar

    # A passagem da folha (mao/pagina em movimento) libera o disparo seguinte.
    transicao = np.full_like(pagina, 35)
    assert not controller.analisar(transicao, agora=2.8).capturar
    outra = _pagina(2)
    cv2.circle(outra, (480, 360), 80, (40, 40, 40), -1)
    assert not controller.analisar(outra, agora=3.0).capturar
    assert not controller.analisar(outra, agora=3.1).capturar
    assert controller.analisar(outra, agora=4.2).capturar


def test_mao_bloqueia_auto_captura_e_vai_para_lista_refazer():
    pagina = _pagina()
    com_mao = pagina.copy()
    # Cor de pele em BGR dentro da faixa usada pelo detector.
    cv2.rectangle(com_mao, (20, 20), (430, 690), (85, 135, 205), -1)

    assert pontuacao_mao(com_mao) >= 0.20
    analise = AutoCaptureController().analisar(com_mao, agora=0.0)
    assert analise.mao_presente
    assert analise.status == "Retire a mao"

    qualidade = avaliar_qualidade(com_mao)
    assert qualidade["repetir_captura"]
    assert "mao ou objeto cobrindo a pagina" in qualidade["motivos_refazer"]


def test_dobra_interna_forte_sem_confundir_pautas():
    pagina = _pagina()
    assert detectar_dobra_grande(pagina)[1] is False

    dobrada = pagina.copy()
    triangulo = np.array([[220, 170], [690, 570], [760, 170]], np.int32)
    cv2.fillConvexPoly(dobrada, triangulo, (170, 170, 170))
    cv2.line(dobrada, (220, 170), (690, 570), (70, 70, 70), 6)
    assert detectar_dobra_grande(dobrada)[1] is True


def test_pagina_cortada_na_borda_nao_dispara():
    cortada = np.full((720, 960, 3), 25, np.uint8)
    cv2.rectangle(cortada, (90, 45), (959, 675), (240, 240, 240), -1)
    cv2.rectangle(cortada, (90, 45), (959, 675), (10, 10, 10), 5)
    for y in range(100, 640, 30):
        cv2.line(cortada, (125, y), (930, y), (60, 60, 60), 2)

    analise = AutoCaptureController().analisar(cortada, agora=0.0)
    assert not analise.enquadrada
    assert not analise.capturar
    assert analise.status == "Afaste ou centralize a pagina"


def test_simulacao_de_lote_captura_cada_pagina_uma_vez_com_mao_e_transicao():
    controller = AutoCaptureController(tempo_estavel=0.4)
    paginas = [_pagina(10), _pagina(11), _pagina(12)]
    for indice, pagina in enumerate(paginas):
        cv2.circle(pagina, (250 + indice * 120, 350), 70, (40, 40, 40), -1)
    capturas = 0
    agora = 0.0
    for pagina_indice, pagina in enumerate(paginas):
        if pagina_indice:
            # A virada real produz um frame intermediário diferente; uma
            # troca apenas da mão, sem este movimento, não libera captura.
            assert controller.analisar(np.full_like(pagina, 35), agora=agora).capturar is False
            agora += 0.1
        # mão/virada entre faces: nunca deve disparar enquanto cobre a folha.
        mao = pagina.copy()
        cv2.rectangle(mao, (0, 0), (400, 719), (85, 135, 205), -1)
        assert controller.analisar(mao, agora=agora).capturar is False
        agora += 0.1
        # vários frames estáveis simulam a câmera; exatamente um dispara.
        for _ in range(8):
            if controller.analisar(pagina, agora=agora).capturar:
                capturas += 1
            agora += 0.1
        # frames parados da mesma folha continuam bloqueados.
        for _ in range(5):
            assert controller.analisar(pagina, agora=agora).capturar is False
            agora += 0.1
    assert capturas == 3


def test_retirar_mao_sem_trocar_folha_nao_libera_duplicata():
    pagina = _pagina(20)
    mao = pagina.copy()
    cv2.rectangle(mao, (0, 0), (400, 719), (85, 135, 205), -1)
    controller = AutoCaptureController(tempo_estavel=0.0)
    assert not controller.analisar(pagina, agora=0.0).capturar
    assert controller.analisar(pagina, agora=0.1).capturar
    assert controller.analisar(mao, agora=0.2).capturar is False
    assert controller.analisar(pagina, agora=0.3).capturar is False
    assert controller.analisar(pagina, agora=0.4).status == "Troque a pagina"


def _pagina_vazia() -> np.ndarray:
    return np.full((720, 960, 3), 30, np.uint8)


def test_matriz_de_transicoes_da_captura():
    """Caracterizacao M1: cada estado alcancavel com status e flag capturar corretos."""
    pagina = _pagina(30)
    controller = AutoCaptureController(tempo_estavel=0.5)

    # SEM_FOLHA: fundo escuro sem pagina detectada.
    sem = controller.analisar(_pagina_vazia(), agora=0.0)
    assert sem.status == "Posicione a pagina"
    assert not sem.capturar and not sem.pagina_presente

    # DETECTADA (estabilizando): pagina presente mas movimento alto no 1o frame.
    detectada = controller.analisar(pagina, agora=0.1)
    assert detectada.pagina_presente
    assert detectada.status in ("Pagina pronta", "Aguarde estabilizar", "Capturada")

    # PRONTA->CAPTURADA com tempo de estabilidade.
    pronta = controller.analisar(pagina, agora=0.2)
    assert pronta.status == "Pagina pronta" if pronta.contagem else pronta.status != "Capturada"
    capturada = controller.analisar(pagina, agora=0.8)
    assert capturada.capturar and capturada.status == "Capturada"

    # COOLDOWN: mesma pagina repetida.
    cooldown = controller.analisar(pagina, agora=0.9)
    assert cooldown.status == "Troque a pagina"
    assert not cooldown.capturar


def test_documento_some_durante_estabilizacao_reseta_contagem():
    pagina = _pagina(31)
    controller = AutoCaptureController(tempo_estavel=0.5)
    controller.analisar(pagina, agora=0.0)
    controller.analisar(pagina, agora=0.3)  # inicia estabilizacao

    some = controller.analisar(_pagina_vazia(), agora=0.45)
    assert some.status == "Posicione a pagina"
    assert not some.capturar

    # A estabilizacao reiniciou: mesmo com tempo decorrido, ainda nao dispara.
    volta = controller.analisar(pagina, agora=0.8)
    assert not volta.capturar
    assert controller.analisar(pagina, agora=1.0).status != "Capturada"
    assert controller.analisar(pagina, agora=1.6).capturar


def test_movimento_forte_durante_estabilizacao_reseta_contagem():
    """Caracterizacao real (M1): a virada (frame quase preto) some a pagina e
    reinicia a estabilizacao; o retorno da pagina passa por Aguarde/Pagina pronta."""
    pagina = _pagina(32)
    controller = AutoCaptureController(tempo_estavel=0.5)
    assert controller.analisar(pagina, agora=0.0).status == "Aguarde estabilizar"
    assert controller.analisar(pagina, agora=0.3).status == "Pagina pronta"

    transicao = np.full_like(pagina, 35)
    assert controller.analisar(transicao, agora=0.35).status == "Posicione a pagina"
    assert controller.analisar(pagina, agora=0.4).status == "Aguarde estabilizar"
    assert controller.analisar(pagina, agora=0.5).status == "Pagina pronta"
    assert controller.analisar(pagina, agora=1.1).capturar


def test_captura_manual_marca_bloqueio_sem_disparo_automatico():
    pagina = _pagina(33)
    controller = AutoCaptureController(tempo_estavel=0.0)
    assert not controller.analisar(pagina, agora=0.0).capturar
    assert controller.analisar(pagina, agora=0.1).capturar

    controller.marcar_capturada(pagina)
    assert controller.analisar(pagina, agora=0.2).status == "Troque a pagina"
    assert not controller.analisar(pagina, agora=0.3).capturar


def test_detector_oscilando_pode_liberar_recaptura_da_mesma_folha():
    """ACHADO M1: um frame sem pagina entre o cooldown destrava o bloqueio; se a
    MESMA folha voltar, a captura e liberada de novo (potencial duplicata).
    Comportamento ATUAL documentado; candidato a correcao no M2."""
    pagina = _pagina(34)
    controller = AutoCaptureController(tempo_estavel=0.0)
    assert controller.analisar(pagina, agora=0.0).status == "Aguarde estabilizar"
    assert controller.analisar(pagina, agora=0.1).capturar
    assert controller.analisar(pagina, agora=0.2).status == "Troque a pagina"

    # Frame sem pagina destrava o bloqueio de cena.
    assert controller.analisar(_pagina_vazia(), agora=0.3).status == "Posicione a pagina"
    # A mesma folha retorna: primeiro frame alto movimento, nao captura.
    assert controller.analisar(pagina, agora=0.4).status == "Aguarde estabilizar"
    assert controller.analisar(pagina, agora=0.4).capturar  # e captura em seguida
