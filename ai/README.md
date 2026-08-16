# AI/ — Loop de engenharia do Digitalizador de Livros

Este diretório é a "memória persistente" do loop de desenvolvimento. Toda task,
decisão e relatório fica aqui. Nada de memória em conversa — memória em arquivo.

## Índice
| Arquivo | Conteúdo |
|---|---|
| `PROJECT_CHARTER.md` | Missão, escopo, limites, métricas |
| `INVARIANTS.md` | Regras absolutas (nunca violar) |
| `ARCHITECTURE.md` | Mapa real do código (M0) |
| `ROADMAP.md` | Milestones M0–M8 adaptados |
| `STATE.json` | Estado atual do loop (fase, tests, next) |
| `LOOP_PROTOCOL.md` | Como tasks são executadas/revisadas |
| `TOKEN_POLICY.md` | Contexto mínimo por task |
| `UX_VALIDATION.md` | Critérios de aceitação de UX |
| `TEST_MATRIX.md` | Mapa dos 77 testes + scripts |
| `DECISIONS.md` | ADRs (D-001…D-009) |
| `tasks/` | Contratos de tarefa (JSON) |
| `prompts/` | Papéis EXECUTOR (Flash) e REVISOR (Pro) |
| `reports/` | baseline, auditoria de dependências, milestones |

## Regra de uso
1. Executor lê: `INVARIANTS` + `tasks/TASK_*.json` + `STATE.json` + arquivos da task.
2. Ao concluir: atualiza `STATE.json`, commit pequeno.
3. Revisor atualiza `DECISIONS.md`/`ROADMAP.md` e escreve `reports/milestone_*.md`.
