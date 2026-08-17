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

## D-010 · Telemetria event-based (M1)
- Eventos JSONL assíncronos (`queue.Queue` + thread de escrita), amostras ~1 Hz,
  allowlist técnica (ids/durações/contagens) e proibição de PII/imagem/texto.
- Desligável via `telemetry.enabled`. Invariante 13. Escrita jamais no caminho crítico.

## D-011 · Achado M1: oscilação do detector pode recapturar a mesma folha
- Caracterização (M1-T03) documentou que um frame sem página durante o cooldown
  destrava o bloqueio de cena; se a MESMA folha voltar, a captura é liberada de novo
  (potencial duplicata). Comportamento atual preservado no M1 (observacional);
  **candidata a correção no M2**. Teste: `test_detector_oscilando_pode_liberar_recaptura_da_mesma_folha`.

## D-012 · Stress da topologia atual (M1-T04)
- Confirmado por medição: fila/lock/retomada com overhead desprezível (~0,3 s/item
  frente a 16–48 s de inferência), RSS estável (~40 MB), kill+retomada sem
  duplicidade/perda, 2 workers bloqueados pelo lock. Topologia NÃO muda.

## D-013 · Estados de captura formalizados em enum (M2)
- `CaptureState` + `FrameAnalysis.estado` em `auto_capture.py`. Valores do enum
  são as mensagens do operador; `status` deriva do enum. Sem mudança de limiares.

## D-014 · Fix da recaptura da mesma folha (M2, achado D-011)
- Cooldown destrava apenas com página nova presente e estável por `tempo_troca`
  (0,6 s). Frame sem página ou mesma folha → permanece bloqueado. Páginas
  sintéticas dos testes agora simulam enquadramento real (diferença por seed).