# Relatório M1 — Baseline operacional (telemetria + caracterização)

> Milestone OBSERVACIONAL: medição, sem otimização/refatoração funcional.
> 2026-08-16 · Suite: **86/86 verdes** (77 baseline + 9 novos).

## 1. Métricas de computação (a função mais lenta ≠ gargalo do operador)

### Fila/worker (stress da topologia atual — 1 worker, OCR/Qwen falsos)
| Escala | Itens | Tempo | Throughput | Backlog máx | RSS início→fim |
|---|---|---|---|---|---|
| 1x | 10 | 5,7 s | 1,75 itens/s | 10 | 38,4→39,6 MB |
| 2x | 20 | 8,5 s | 2,37 itens/s | 20 | 39,7→40,1 MB |
| 5x | 50 | 17,2 s | 2,92 itens/s | 50 | 39,7→39,6 MB |
| 10x | 100 | 32,4 s | 3,08 itens/s | 100 | 39,7→39,7 MB |

- RSS **estável** (~40 MB) — a fila não acumula memória com a carga.
- Overhead do maquinário ≈ 0,3 s/item; **desprezível** frente ao OCR/Qwen reais (16,5–48 s/item).
- Backlog só aparece na fase Qwen (1 a 1, por projeto) e cresce até o total da fila — esperado e saudável.
- Interrupção+retomada (20 itens, corte a 60%): fase 1 `pausado` com 12 concluídos; fase 2 `concluido`; **0 duplicados, 0 perdidos, 20 execuções rápidas + 20 Qwen (1 por registro)**, 0 pendentes restantes.
- 2 workers no mesmo banco: segundo **bloqueado** (`já existe um trabalhador de nomes ativo`) — topologia preservada.

### Captura (medido sintético — câmera real pendente de sessão do operador)
| Métrica | 720×960 | 1920×1080 |
|---|---|---|
| Latência detector (`analisar`) média | 16,7 ms | 19,2 ms |
| Latência detector p95 | 18,4 ms | 23,0 ms |
| Gravação JPG Q95 média | 2,5 ms | 7,3 ms |
| Ritmo simulado até captura (estável 1,2 s) | 1,43 s | — |

- Detector **bem dentro** do orçamento do timer de 50 ms (≈20 fps de folga).
- `capture.sample` (FPS real, jitter da UI) será coletado na próxima sessão com câmera.

## 2. Métricas de operação (gargalo para o OPERADOR)

| Pergunta | Resposta M1 |
|---|---|
| Tempo até a próxima captura | ~1,4 s + ritmo humano de virar a folha (detector não é o limite) |
| Freeze da interface | Não medido sem câmera real → eventos `capture.sample` prontos |
| Capturas/min | Limitado pelo operador, não pelo sistema |
| Falso disparo | **ACHADO M1-T03**: oscilação do detector pode destravar o cooldown e **recapturar a mesma folha** (duplicata potencial) |
| Espera desnecessária | Operador não vê o estado da fila; só o contador de revisões |
| Backlog sem recuperação | Não observado (fila retoma e conclui em todas as escalas) |

## 3. Gargalos reais
1. **Qwen em CPU** (16,5 s nome / 48,5 s assento) — já conhecido; desacoplado da captura por design (não trava o operador).
2. **Reflexo/glare não detectado** (gap confirmado) — pode gerar foto ruim silenciosa; recaptura só por sorte.
3. **Recaptura da mesma folha por oscilação do detector** (achado novo, ver acima).
4. **Sem visão de "o que falta no livro"** (recapturas/revisões espalhadas).

## 4. O que NÃO precisa ser mexido (confirmado por medição)
- Maquinário da fila/lock/retomada — overhead desprezível, topologia correta.
- Detector de página (16–23 ms) e gravação JPG (2,5–7,3 ms).
- Memória do worker (~40 MB estável).
- OCR rápido / arquitetura de 2 etapas.

## 5. Candidatos para M2/M3 (recomendações)
- **M2**: formalizar estados em enum; tratar a **oscilação do detector** (não destravar cooldown com frame sem página que não seja uma troca real); HUD de estado.
- **M3**: glare/reflexo (novo); benchmark mão-heurística vs MediaPipe com frames reais.
- **M4/M5**: painel fim-de-livro + recaptura por folha com motivo (resolve o gargalo "não sei o que falta").

## 6. Entregas M1
- `src/services/telemetry.py` (eventos JSONL assíncronos, sem PII, desligável) + hooks (worker, pipeline, câmera, scan_screen) + ativação nos 2 entrypoints.
- 4 testes de telemetria (eventos, sem PII, desligável, amostrador).
- 5 testes de caracterização da captura (matriz + borda) — sem refatorar `auto_capture.py`.
- `scripts/stress_fila_lote.py`, `scripts/medir_captura_sintetica.py` e `.tmp_*_result.json`.
- Suite **86/86**.

## 7. Próximo passo
Aguardando aprovação para iniciar **M2** (tasks propostas acima). PARADA para revisão conforme protocolo.
