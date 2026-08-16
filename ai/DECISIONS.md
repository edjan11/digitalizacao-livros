# DECISIONS — Registro de decisões (ADR leve)

> Cada decisão relevante fica registrada aqui com contexto, alternativa e resultado.
> Atualizado sempre que uma task muda um comportamento.

## D-001 · Manter Qwen2-VL-2B local como motor de manuscrito
- **Contexto**: CPU-only; 16,5 s/nome; candidatos testados.
- **Alternativas**: GOT-OCR2 (0/19, 18,5%, 34,6 s), Qwen2.5-VL-3B via Ollama (5/19,
  78%, 42 s), Kraken (0/19, 21,7%).
- **Resultado**: nenhum superou; manter. Benchmarks em `PROCESSAMENTO_A07.md`.

## D-002 · `qwen_max_pixels` permanece 301056 (qualidade)
- 113k px acelera ~40% na linha de nome (9,7 s) com 86→81% de acerto; no assento
  inteiro degrada campos (termo/data). Chave de configuração disponível; não muda
  o padrão.

## D-003 · Exibição retificada na Consulta/Revisor
- Aplicar `retificar_formulario` em memória na miniatura e no revisor; aceitar
  qualquer `applied=True` (garante ao menos alinhamento horizontal).
- Validação: 637 fotos do A-07 → 590 exibem retificadas; 36 já retas; 11 capas/índices.

## D-004 · Moldura do assento == recorte do OCR
- `ImageViewer.set_highlight` deriva de `bbox_registro(indice,total)`; faixa
  ampliada para (0.045, ·, 0.76, ·) com margens maiores. O que se vê é o que se lê.

## D-005 · Evidência de termo na UI
- Mostrar `ocr_termo/termo_status/confianca_termo` (Consulta + Revisor). A
  sequência auditada continua sendo a decisão; o OCR vira evidência visível.

## D-006 · Git baseline (M0)
- `git init` + `.gitignore` (venv/build/dist/models/.tmp/logs) + commit `1b7266d`.
- Motivo: proteção contra regressões e rastreio de tasks.

## D-007 · Nenhuma dependência instalada no M0
- mediapipe/pytest-qt/pytest-timeout/ruff/psutil **ausentes** → decisão de adiar
  cada uma está em `ai/reports/dependency_audit.md`. Nada instalado sem provar.

## D-008 · Não tocar OrganizadorFirmas
- Escopo desta rodada é apenas `digitalizacao-livros`.

## D-009 · Loops de LLM externos (Deep Code CLI) adiados
- Plano avaliado (documentação oficial confirmada), mas a execução ocorre via
  opencode nesta sessão — mesmo protocolo, zero instalação/custo.