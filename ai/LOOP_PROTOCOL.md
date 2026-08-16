# LOOP_PROTOCOL — Como as tasks são executadas e revisadas

> Executado por opencode nesta sessão (papel duplo executor/revisor, com
> disciplina de papéis). O protocolo é o mesmo que o plano "Deep Code Pro/Flash".

## Papéis
- **Executor (Flash)**: implementa tasks pequenas (1 mudança lógica, 1–4 arquivos),
  roda os testes-alvo, no máximo **2 ciclos de reparo** por falha.
- **Revisor (Pro)**: define milestones, contratos, investiga regressões difíceis,
  decide arquitetura, faz revisão final.

## Ciclo de uma task
```
task em ai/tasks/TASK_*.json
   → executor implementa (invariantes + task + arquivos relevantes)
   → roda teste-alvo
   → OK? → commit pequeno + atualiza STATE.json
   → ERRO? → reparo #1 → reparo #2 → ainda erro? → ESCALA para revisor
```

## Regra de escalonamento imediato para o revisor (não espera 2 reparos)
- migração de banco/schema
- threading complicado, lock/fila
- perda de dados, hash do original
- alteração de arquitetura
- PyInstaller/spec
- segurança/privacidade (segredos, envio externo)
- regressão grande (suite)

## Critérios de aceitação (toda task DEVE ter)
1. Objetivo único e verificável.
2. Arquivos permitidos (1–4).
3. Critérios de aceitação mensuráveis.
4. Comando(s) de teste.
5. Risco e rollback.
6. Orçamento de contexto.

## Commit
- 1 commit por task, mensagem no padrão do repo.
- Nunca commit de secrets/`config.yaml` com senha.
- Verificar `git status`/`git diff` antes de commitar.

## A cada 3–5 tasks
- Rodar `pytest -q` completo.
- Atualizar `ai/STATE.json`, `ai/DECISIONS.md` e `ai/reports/baseline.md` se mudou.

## Fim de milestone
- Suite completa verde.
- `ai/reports/milestone_<M>.md` com o que mudou, métricas e riscos.
- Apresentar ao operador antes de iniciar o próximo milestone.