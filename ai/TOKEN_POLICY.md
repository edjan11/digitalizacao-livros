# TOKEN_POLICY — Contexto mínimo por chamada/task

Objetivo: nunca enviar o repositório inteiro nem "a novela completa" a cada task.
Contexto de execução é sempre composto por:

```
ai/INVARIANTS.md          (regras — estável)
ai/tasks/TASK_<ID>.json   (contrato daquela task — varia)
ai/STATE.json             (estado resumido — varia pouco)
arquivos relevantes       (só os listados na task)
teste/erro relevante      (a última saída do teste-alvo)
```

## O que NUNCA entra no contexto
- Conversa completa anterior.
- Logs completos (apenas o erro/trecho).
- Reasoning de chamadas anteriores.
- Árvore inteira de arquivos não relacionados.
- Dumps de banco.

## Estabilidade para cache
- Prefixo do prompt (invariantes + charter) permanece idêntico entre tasks.
- Varia apenas: TASK + estado + arquivos + erro.
- Resultado: maior chance de cache hit e menos tokens repetidos.

## Orçamento por task
- Objetivo e arquivos cabem em poucas centenas de tokens.
- Se a task exigir ler > 6 arquivos grandes, ela está grande demais → dividir.