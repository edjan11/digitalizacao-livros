---
name: loop-engineer
description: Executa uma task do loop de engenharia (M0–M8) seguindo ai/LOOP_PROTOCOL.md: contexto mínimo, teste-alvo, máx. 2 reparos, escalonamento, commit pequeno.
---

# Skill: loop-engineer

## Quando usar
Quando uma task em `ai/tasks/` precisa ser implementada ou um milestone deve avançar.

## Fluxo
1. Leia `ai/INVARIANTS.md` e `ai/STATE.json`.
2. Leia a task `ai/tasks/TASK_<ID>.json` fornecida.
3. Implemente apenas o objetivo, respeitando `allowed_files` (1–4 arquivos).
4. Rode `test_commands` da task.
5. Falhou? Reparo #1 → Reparo #2 → ainda falhou? **ESCALE para o revisor**
   (prompt em `ai/prompts/REVIEWER_PRO.md`) com erro + tentativas.
6. Passou? Atualize `ai/STATE.json` e faça um commit pequeno.
7. A cada 3–5 tasks: `pytest -q` completo e atualize `ai/reports/baseline.md` se necessário.

## Regras
- Invariantes têm prioridade máxima (`ai/INVARIANTS.md`).
- Não instale bibliotecas. Não envie imagens para fora. Não toque no OrganizadorFirmas.
- 1 commit por task. Verifique `git status` antes de commitar.
