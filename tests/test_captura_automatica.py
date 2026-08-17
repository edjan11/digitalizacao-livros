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
    """Folha sintetica realista (M2-T02): o enquadramento varia com o seed,
    como frames reais de camera — paginas diferentes diferem em muito mais
    que a caligrafia, permitindo o discriminador mudanca_pagina funcionar.
    O papel fica em ~225 (como o p99 das fotos reais do A-07), deixando o
    reflexo (>=248) claramente separado para o detector de glare (M3-T01)."""
    image = np.full((720, 960, 3), 215 + (seed % 4) * 6, np.uint8)
    dx = (seed % 5) * 14
    dy = (seed % 3) * 12
    x0, y0 = 90 + dx, 45 + dy
    cv2.rectangle(image, (x0, y0), (870 + dx, 675 + dy), (225, 225, 225), -1)
    cv2.rectangle(image, (x0, y0), (870 + dx, 675 + dy), (25, 25, 25), 5)
    for y in range(y0 + 55, y0 + 600, 30):
        cv2.line(image, (x0 + 35, y), (x0 + 745, y), (65, 65, 65), 2)
    cv2.putText(
        image, f"Numero {6800 + seed}", (x0 + 45, y0 + 45),
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
    # Pagina nova: confirmacao de troca (tempo_troca) destrava no proximo call
    # (aqui em 4.7) e a estabilidade de 1,0 s comeca nesse instante.
    assert not controller.analisar(outra, agora=4.7).capturar
    assert controller.analisar(outra, agora=5.7).capturar


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
        # Frames suficientes para: confirmacao de troca (tempo_troca) + estabilidade.
        for _ in range(16):
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


def test_estado_enum_consistente_com_status():
    from src.capture.auto_capture import CaptureState

    pagina = _pagina(50)
    controller = AutoCaptureController()
    estados_vistos = set()
    for t in (0.0, 0.1, 0.2, 0.3):
        resultado = controller.analisar(pagina, agora=t)
        assert resultado.estado is not None
        assert resultado.status == resultado.estado.value
        estados_vistos.add(resultado.estado)
    assert CaptureState.AGUARDANDO_ESTABILIDADE in estados_vistos
    assert CaptureState.PAGINA_PRONTAA in estados_vistos


def _com_reflexo(pagina: np.ndarray, cobertura: float = 0.06) -> np.ndarray:
    """Adiciona um reflexo especular (blob claro e lavado) sobre a pagina."""
    com_reflexo = pagina.copy()
    h, w = com_reflexo.shape[:2]
    cx, cy = int(w * 0.45), int(h * 0.40)
    raio = int(np.sqrt(w * h * cobertura / np.pi))
    mascara = np.zeros((h, w), np.float32)
    cv2.circle(mascara, (cx, cy), raio, 1.0, -1)
    mascara = cv2.GaussianBlur(mascara, (0, 0), sigmaX=raio * 0.35)
    blob = np.full((h, w, 3), 255, np.uint8)
    com_reflexo = np.where(
        mascara[..., None] > 0.35, blob, com_reflexo
    ).astype(np.uint8)
    return com_reflexo


def test_detectar_glare_identifica_reflexo_e_ignora_pagina_normal():
    from src.imaging.quality import avaliar_qualidade, detectar_glare

    pagina = _pagina(60)
    grau, status = detectar_glare(pagina)
    assert status == "ok"
    assert avaliar_qualidade(pagina)["reflexo_status"] == "ok"

    com_reflexo = _com_reflexo(pagina, cobertura=0.06)
    grau, status = detectar_glare(com_reflexo)
    assert status == "reflexo_forte"
    assert grau >= 0.03

    qualidade = avaliar_qualidade(com_reflexo)
    assert qualidade["reflexo_status"] == "reflexo_forte"
    assert qualidade["repetir_captura"]
    assert "reflexo forte sobre a pagina" in qualidade["motivos_refazer"]


def test_detectar_glare_aviso_para_reflexo_pequeno():
    from src.imaging.quality import avaliar_qualidade, detectar_glare

    pagina = _pagina(61)
    com_reflexo_pequeno = _com_reflexo(pagina, cobertura=0.012)
    grau, status = detectar_glare(com_reflexo_pequeno)
    assert status == "aviso"
    assert avaliar_qualidade(com_reflexo_pequeno)["reflexo_status"] == "aviso"
    # Aviso nao força recaptura.
    assert not avaliar_qualidade(com_reflexo_pequeno)["repetir_captura"]


def test_hud_da_camera_colore_estado_e_mostra_bloqueio():
    """M2-T03: HUD offscreen — cor por estado, contagem e indicador de bloqueio."""
    from PySide6.QtWidgets import QApplication
    from src.capture.auto_capture import CaptureState, FrameAnalysis
    from src.ui.camera_capture_dialog import CameraCaptureDialog, CORES_ESTADO

    app = QApplication.instance() or QApplication([])
    dialogo = CameraCaptureDialog(
        capture_dir=Path(__file__).resolve().parent / "hud_inexistente",
        context_provider=lambda: "ctx",
        parent=None,
    )
    frame = np.full((200, 300, 3), 220, np.uint8)

    def analise(estado, contagem=None):
        return FrameAnalysis(
            movimento=0.0, foco=200.0, mao_presente=False,
            pagina_presente=True, status=estado.value,
            contagem=contagem, capturar=False, enquadrada=True,
            pagina_contorno=None, estado=estado,
        )

    dialogo._mostrar_preview(frame, analise(CaptureState.PAGINA_PRONTAA, contagem=0.8))
    texto_pronto = dialogo.status.text()
    assert "Pagina pronta" in texto_pronto
    assert "0.8" in texto_pronto
    assert CORES_ESTADO[CaptureState.PAGINA_PRONTAA] in dialogo.status.styleSheet()

    dialogo._mostrar_preview(frame, analise(CaptureState.TROQUE_PAGINA))
    assert "BLOQUEADO" in dialogo.status.text()
    assert CORES_ESTADO[CaptureState.TROQUE_PAGINA] in dialogo.status.styleSheet()

    dialogo._mostrar_preview(frame, analise(CaptureState.SEM_FOLHA))
    assert "Posicione a pagina" in dialogo.status.text()
    assert CORES_ESTADO[CaptureState.SEM_FOLHA] in dialogo.status.styleSheet()
    dialogo.close()


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


def test_detector_oscilando_nao_recaptura_a_mesma_folha():
    """Correcao M2-T02 (D-011): a mesma folha que volta NAO destrava o cooldown.

    Um frame sem pagina ou a mesma folha retornando nao podem liberar recaptura.
    """
    pagina = _pagina(34)
    controller = AutoCaptureController(tempo_estavel=0.0)
    assert controller.analisar(pagina, agora=0.0).status == "Aguarde estabilizar"
    assert controller.analisar(pagina, agora=0.1).capturar
    assert controller.analisar(pagina, agora=0.2).status == "Troque a pagina"

    # Frame sem pagina permanece bloqueado (antes destravava o cooldown).
    sem = controller.analisar(_pagina_vazia(), agora=0.3)
    assert sem.status == "Troque a pagina"
    assert not sem.capturar
    # A mesma folha voltando continua bloqueada por tempo indeterminado.
    for t in (0.4, 0.5, 0.8, 1.2):
        repetida = controller.analisar(pagina, agora=t)
        assert repetida.status == "Troque a pagina"
        assert not repetida.capturar


def test_cooldown_destrava_com_pagina_nova_estavel_por_tempo_troca():
    """Pagina NOVA, presente e estavel por tempo_troca destrava o cooldown."""
    pagina1 = _pagina(40)
    pagina2 = _pagina(41)
    controller = AutoCaptureController(tempo_estavel=0.3, tempo_troca=0.5)
    assert controller.analisar(pagina1, agora=0.0).status == "Aguarde estabilizar"
    assert controller.analisar(pagina1, agora=1.0).status == "Pagina pronta"
    assert controller.analisar(pagina1, agora=1.3).capturar

    # Virada: sem pagina -> bloqueado; pagina nova entra.
    assert controller.analisar(_pagina_vazia(), agora=1.4).status == "Troque a pagina"
    assert controller.analisar(pagina2, agora=1.5).status == "Troque a pagina"  # 1o frame
    assert controller.analisar(pagina2, agora=1.6).status == "Troque a pagina"  # inicia troca
    assert controller.analisar(pagina2, agora=1.9).status == "Troque a pagina"  # confirmando
    destrave = controller.analisar(pagina2, agora=2.2)  # 0,6 s depois do inicio
    assert destrave.status == "Pagina pronta"
    assert not controller.analisar(pagina1, agora=2.5).capturar
