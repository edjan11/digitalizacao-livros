# ROADMAP — Milestones M0–M8 (adaptado ao que já existe)

> Coluna "Já existe" indica o quanto o milestone está coberto pelo código atual.
> Nada é reconstruído; o que falta é formalizar, preencher gap ou testar.

| # | Milestone | Já existe | O que falta |
|---|---|---|---|
| **M0** | Baseline + inventário | 77/77 verdes, docs, git criado | (concluído 2026-08-16) |
| **M1** | Telemetria + caracterização | — | **Observacional (sem otimizar/refatorar)**: T00 dev-deps (psutil, pytest-timeout); T01 telemetria do worker/OCR (eventos, 1 Hz, sem PII); T02 telemetria da captura (FPS, latência detector, transições, salvar, filas, freeze UI); T03 matriz de transições da state machine existente (sem refatorar); T04 stress da topologia atual (1 worker, 1x/2x/5x/10x, kill+retomada, duplicidade/perda; 2 workers só em banco isolado); T05 relatório operacional (gargalo para o OPERADOR, não só o mais lento) |
| **M2** | Capture State Machine | ~85% (`AutoCaptureController` com 6 testes) | Formalizar estados em `enum`; HUD de estado na prévia; mensagens consistentes; testes de transição |
| **M3** | Mão/oclusão + Quality Gate | Mão heurística (`pontuacao_mao`) | Benchmark heurística vs MediaPipe em frames reais (só instala se vencer); **detecção de reflexo/glare** (novo, OpenCV) |
| **M4** | Quality Audit background | Já em background (filas `scan_screen.py`) | Glare; **resumo fim-de-livro** (aprovadas/revisar/recapturar por motivo); persistência de qualidade por imagem |
| **M5** | Fluxo de recaptura | Duplicidade resolve; `faces_faltantes` | Painel RECAPTURAR: lista de folhas com motivo, botões ABRIR FOTO / RECAPTURAR, integração com sessão |
| **M6** | UX do digitalizador | Tela atual funcional | HUD de estados, atalhos (ESPAÇO = captura manual), contadores ao vivo, mensagens orientadas a ação |
| **M7** | Consulta/busca | Filtros + fuzzy + API | Navegação Explorer (2 painéis, pastas recolhíveis, ~25 linhas); avaliar **FTS5** antes de embeddings; busca rápida por folha/termo |
| **M8** | Resiliência + Release | PyInstaller ok, fila retomável | Smoke test automatizado do `.exe`; verificação de hash do original no build; guardar versões |

## Regras de progresso
- Fases são concluídas apenas com suite verde **e** critério de aceitação explícito.
- M0 já entregou: git baseline, inventário, auditoria de dependências, este roadmap.
- Ordem recomendada: M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8. M7 pode ser puxado antes se a prioridade do operador for busca.