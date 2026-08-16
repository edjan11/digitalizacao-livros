# UX_VALIDATION — Critérios de aceitação de UX

> Critérios para validar mudanças de interface SEM depender de olho humano a cada
> iteração: testável em modo offscreen (`QT_QPA_PLATFORM=offscreen`).

## Princípios
1. **Estado visível**: o operador sempre sabe o que o sistema está fazendo
   (capturando, processando, revisando) — sem adivinhar.
2. **Ação orientada**: mensagens dizem o que FAZER ("Retire a mão", "Afaste a
   página", "Troque a página"), não só o que está errado.
3. **Nada de memória**: se o livro foi fechado no meio, ao reabrir o operador vê
   o que falta (recapturas, revisões) — não "acho que ficou algo pra trás".
4. **Zero confirmação falsa**: sugestão nunca parece confirmada (cor/ícone/classe).
5. **Fluidez**: captura não espera OCR/Qwen; resultados chegam e atualizam a UI.

## Critérios verificáveis (por milestone)
- **M2/M6 (HUD captura)**: os estados (SEM_FOLHA, RETIRE_MAO, AFASE, AGUARDANDO_FOCO,
  AGUARDE_ESTABILIZAR, PAGINA_PRONTA, CAPTURADA) aparecem como texto/ícone na prévia
  e mudam conforme `FrameAnalysis.status`.
- **M5 (recaptura)**: painel lista `folha + face + termos + motivo` e tem
  `ABRIR FOTO` / `RECAPTURAR`; clicar recaptura abre a câmera apontando para aquela folha.
- **M7 (consulta Explorer)**: 2 painéis (árvore de livros + lista de folhas/termos);
  pastas recolhíveis; máx. ~25 linhas visíveis; digitar folha/termo navega direto.
- **M4 (fim de livro)**: resumo com `Aprovadas / Revisar / Recapturar (motivo)`.

## Como testar (sem câmera real)
- `AutoCaptureController` com frames sintéticos (testes existentes já cobrem).
- UI: `QApplication` offscreen + asserts em textos/estados de widgets.
- Nunca depender de timing real; injetar `agora` e frames.