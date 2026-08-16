---
name: ux-auditor
description: Audita mudanças de interface do Digitalizador/Consulta contra ai/UX_VALIDATION.md usando testes offscreen e critérios verificáveis, sem depender de olho humano.
---

# Skill: ux-auditor

## Quando usar
Antes de fechar qualquer task que mexa em UI (captura, revisor, consulta, painéis).

## Fluxo
1. Leia `ai/UX_VALIDATION.md` e os critérios do milestone da task.
2. Verifique se a mudança é testável em `QT_QPA_PLATFORM=offscreen`
   (estados de widgets, textos, visibilidade).
3. Rode a suite UI: `tests/test_modelos_visuais.py` + arquivos relacionados.
4. Confira os princípios: estado visível, ação orientada, nada de memória,
   zero confirmação falsa, fluidez (captura não espera OCR).
5. Reporte: critérios atendidos/não atendidos + evidência (textos de teste).

## Regras
- Não usar screenshots humanos como critério único.
- Qualquer estado de UI novo precisa de um teste que o verifique.
