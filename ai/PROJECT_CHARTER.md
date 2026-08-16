# PROJECT CHARTER — Digitalizador de Livros / ConsultaAcervo

> Sistema EM PRODUÇÃO. Este documento define missão, escopo, limites e métricas de sucesso.
> Toda tarefa do loop deve respeitar este charter.

## Missão
Digitalizar, indexar e consultar livros de registro civil preservando o original
(300 DPI), usando OCR/IA **apenas como sugestão** e mantendo auditoria total do
que foi lido, corrigido e gravado.

## Escopo desta rodada
- **Somente o repositório `D:\T.I\digitalizacao-livros`** (Digitalizador + ConsultaAcervo + API + fila persistente).
- **O projeto `D:\T.I\fichasFirmas` (OrganizadorFirmas) é intocado.**

## O que NÃO é objetivo (limites)
- Não reescrever o sistema.
- Não trocar motores OCR sem benchmark reproduzível.
- Não introduzir LangChain/LangGraph/Redis/Celery/RabbitMQ/Kafka ou novo banco sem ADR demonstrando necessidade.
- Não enviar imagens do acervo a serviços externos sem autorização explícita.
- Não instalar bibliotecas antes de `ai/reports/dependency_audit.md` provar necessidade.

## Métricas de sucesso
1. 77/77 testes verdes como baseline; **toda task termina com suite verde** (ou com falha documentada como regressão).
2. Captura continua **independente da conclusão do OCR/Qwen** (nunca bloquear a câmera por IA).
3. JPG original imutável (hash antes/depois em qualquer operação).
4. Fila persistente e retomável; um único worker por banco.
5. UX: operador termina um livro sem depender de memória (recaptura por folha, resumo fim-de-livro).

## Estado
- Baseline: `77 passed in 363.62s` (2026-08-16)
- Commit inicial: `1b7266d`
- Fase atual: M0 concluído