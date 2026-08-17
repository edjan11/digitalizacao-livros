# Relatório M3 — Quality Gate: glare + decisão de mãos

> 2026-08-16 · Suite: **92/92 verdes** (90 no início do M3).

## 1. M3-T01 — Detecção de reflexo/glare (novo, OpenCV)

**Algoritmo** (`detectar_glare` em `src/imaging/quality.py`):
- Blob compacto e "lavado": pixels ≥ 248 (papel real do A-07 fica em 207–228 no
  p99) com desvio local < 12 (janela 15×15) formando um componente conexo grande.
- `reflexo_forte` (≥ 3% da imagem e < 95%) → entra em `motivos_refazer` →
  `repetir_captura = True` (foto ruim não passa silenciosa).
- `aviso` (≥ 1%) → apenas status, sem forçar recaptura.

**Validação:**
- **637/637 fotos reais do A-07 sem falso positivo** (blob 0,0 em todas).
- Sintético (blob gaussiano 255 sobre página): forte → refazer; pequeno → aviso.
- Custo: ~174 ms na foto cheia; no pipeline roda na cópia 1200px (bem menor).

## 2. M3-T02 — Benchmark de mãos (heurística vs MediaPipe)

MediaPipe 1.0.1 instalado **somente para o experimento** e **desinstalado após**.

| Cenário | Esperado | Heurística | MediaPipe |
|---|---|---|---|
| sem_mao | ausente | ✓ | ✓ |
| mao_clara/escura 30% | presente | ✓ (3/3) | ✗ (0/3) |
| punho 20% | presente | ✓ | ✗ |
| mao_parcial_borda | presente | ✓ | ✗ |
| dedos_na_borda | presente | ✗ | ✗ |
| mao 50% | presente | ✓ (2/2) | ✗ (0/2) |

- **Acertos: heurística 8/9 · MediaPipe 1/9.**
- **Custo: heurística 8,2 ms/frame · MediaPipe 24,7 ms/frame** (3× maior).
- Os blocos de pele não são "mãos anatômicas" para o MediaPipe — o benchmark
  sintético não favorece o MP. Mas também **não há evidência real** de que o MP
  supere a heurística no cenário do operador (mão grande cobrindo a folha).

**Decisão (D-016):** manter a heurística (`pontuacao_mao`). MediaPipe **não entra**
em produção. Reavaliar SOMENTE se o operador reportar falsos negativos reais
(mãos finas/dedos) — nesse caso, capturar frames reais com mão e re-benchmarkar
(`scripts/benchmark_mao.py` pronto, reinstalação documentada).

## 3. Testes novos
- `test_detectar_glare_identifica_reflexo_e_ignora_pagina_normal`
- `test_detectar_glare_aviso_para_reflexo_pequeno`
- `scripts/analisar_glare_real.py` (validação das 637 fotos, reproduzível)
- `scripts/benchmark_mao.py` + `.tmp_benchmark_mao.json`

## 4. Riscos restantes (para M4/M5)
1. FPS/jitter reais da câmera ainda não coletados (sessão do operador pendente).
2. Mão heurística pode falhar com tons de pele extremos ou dedos finos —
   mitigação documentada (reavaliação com frames reais).
3. Glare com cobertura > 95% vira "aviso" (não força recaptura) — deliberado.

## 5. Recomendação
- **M4**: painel "fim de livro" (aprovadas/revisar/recapturar por motivo) usando
  os dados de qualidade já persistidos; glare agora alimenta `precisa_revisao`.
- PARADA para revisão conforme protocolo.
