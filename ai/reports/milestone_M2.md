# Relatório M2 — Estados de captura formalizados + fix D-011 + HUD

> 2026-08-16 · Suite: **90/90 verdes** (89 no início do M2, 90 ao fechar).

## 1. O que mudou

### M2-T01 — `CaptureState` (enum)
- `src/capture/auto_capture.py`: novo enum `CaptureState` com os 9 estados
  (SEM_FOLHA, MAO_PRESENTE, NAO_ENQUADRADA, AGUARDANDO_FOCO,
  AGUARDANDO_ESTABILIDADE, PAGINA_PRONTAA, CAPTURADA, TROQUE_PAGINA,
  CAPTURA_MANUAL); valores = mensagens exibidas.
- `FrameAnalysis.estado: CaptureState | None` tipado; `status` (str) derivado
  do enum. Sem mudança de limiares nem de lógica (fora o fix T02).

### M2-T02 — Fix do achado D-011 (recaptura da mesma folha)
- **Antes:** qualquer mudança grande de cena (inclusive um frame sem página —
  flicker/glare) destravava o cooldown; a mesma folha voltando era recapturada.
- **Depois:** o cooldown só destrava com **página presente, diferente da
  capturada (mudança ≥ `mudanca_pagina`) e estável por `tempo_troca` (0,6 s)**.
  Frame sem página ou a mesma folha → permanece `TROQUE_PAGINA`.
- Novo parâmetro `tempo_troca` (default 0,6 s) no `AutoCaptureController`.
- Páginas sintéticas dos testes agora simulam o enquadramento real (deslocamento
  e brilho por seed), para o discriminador funcionar como em fotos reais
  (páginas diferentes: diff média 36 vs mesma folha: 0).

### M2-T03 — HUD da câmera
- Rótulo de status colorido por estado (verde pronto, laranja espera, vermelho
  bloqueio/mão, azul capturada).
- Overlay na prévia: faixa superior com estado + contagem regressiva e
  indicador `[BLOQUEADO]`.
- Testável offscreen (teste cobre texto, contagem e cores).

## 2. Testes (novos/alterados em `tests/test_captura_automatica.py`)
- `test_estado_enum_consistente_com_status` — enum ≡ status.
- `test_detector_oscilando_nao_recaptura_a_mesma_folha` — comportamento corrigido.
- `test_cooldown_destrava_com_pagina_nova_estavel_por_tempo_troca` — troca real.
- `test_hud_da_camera_colore_estado_e_mostra_bloqueio` — HUD offscreen.
- Ajustes de timing: `test_captura_so_depois_de_estavel...` e
  `test_simulacao_de_lote...` (cooldown agora inclui `tempo_troca`).

## 3. Riscos restantes (para M3+)
1. A troca só é confirmada por diffs média (`mudanca_pagina=10`): uma folha
   nova MUITO parecida em enquadramento idêntico poderia ficar bloqueada
   (falso negativo) — improvável em frames reais (diff real 36+).
2. Mesma folha recolocada em posição bem diferente pode passar por "folha nova"
   → duplicata potencial; mitigada a jusante pelo detector de duplicidade
   (confirmação do operador).
3. FPS/jitter reais da câmera continuam pendentes de sessão do operador
   (telemetria pronta: `capture.sample`).

## 4. Estado
- Suite: 90/90 (7 min).
- Git: commits `bea4987` (T01/T02) e `17ddac1` (T03).
- `tempo_troca` exposto no construtor para ajuste fino sem rebuild se necessário.

## 5. Recomendação
- **M3**: glare/reflexo (OpenCV) + benchmark de mão heurística vs MediaPipe com
  frames reais; coletar `capture.sample` numa sessão real do operador primeiro.
- PARADA para revisão conforme protocolo.
