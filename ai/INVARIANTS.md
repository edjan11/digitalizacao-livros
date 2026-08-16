# INVARIANTS — Regras absolutas do projeto

Estas regras NUNCA podem ser violadas por uma task. Uma task que quebra um
invariante é regressão, independente de passar nos testes.

1. **O JPG original é imutável.** Nenhuma rotação, retificação, moldura,
   recorte ou marcação pode sobrescrever o arquivo armazenado. Operações usam
   cópia em memória ou derivados (`.tmp`, `%TEMP%`).
2. **OCR/IA apenas sugere; nunca confirma sozinho.** Sugestão exige revisão
   humana. Apenas confirmação/correção do operador torna o dado verde/ativo.
3. **Captura independente da conclusão do OCR.** A câmera continua funcionando
   enquanto hashes, qualidade, duplicidade, OCR rápido e Qwen rodam em segundo
   plano (threads próprias, filas).
4. **Fila persistente e retomável.** Fechar o app no meio do livro deve retomar
   do ponto exato ao reabrir, sem reprocessar e sem duplicar.
5. **Um único worker por banco.** Lock exclusivo (`*nomes.lock`) impede duas
   instâncias consumindo o mesmo lote.
6. **Nenhuma inferência pesada na thread principal da UI.** Qwen/OCR sempre em
   QThread/processo separado.
7. **API permanece somente leitura.** Nada de escrita via HTTP.
8. **A resposta atrasada do Qwen nunca é salva em outro termo.** Validação por
   hash + posição + contexto imutável antes de persistir.
9. **Tudo pesquisável tem rastreabilidade.** Valor, confiança, motor, fonte,
   status, caixa e geometria ficam registrados.
10. **Não toque no OrganizadorFirmas** nesta rodada.
11. **Regressão = tarefa não concluída.** `76 passed 1 failed` não é "quase
    ok"; é falha a ser corrigida antes de avançar.
12. **Não descartar alterações locais do usuário.** Antes de sobrescrever
    arquivos, conferir `git status`/`git diff`.
13. **TELEMETRIA NÃO PODE ALTERAR O COMPORTAMENTO MEDIDO.**
    - desligável (`telemetry.enabled`);
    - sem imagem no log;
    - sem OCR/texto sensível/dados pessoais por padrão (apenas identificadores
      técnicos: ids, durações, contagens, flags);
    - sem log por frame (eventos de transição + amostras ~1 Hz);
    - sem I/O síncrono pesado no caminho da captura (fila de logging +
      thread de escrita).

## Escala de gravidade de falha
- Quebra invariante → bloqueio imediato do milestone, escalar para o revisor.
- Teste falhou → corrigir (executor máx. 2 reparos), senão escalar.
- Degradação sem teste (ex.: 86%→81% de acerto) → ADR em `ai/DECISIONS.md`.