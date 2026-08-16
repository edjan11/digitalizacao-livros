# Arquitetura recomendada

## Decisão principal

O processamento continua em Python, porque já concentra OpenCV, OCR, fila de
trabalho, banco e preservação dos originais. O balcão dos escreventes não deve
carregar Qwen, Tesseract ou PyTorch: ele deve apenas consultar a API e carregar
fotos.

```text
Digitalizador (PySide6 + OpenCV)
  captura, qualidade, duplicidade, OCR rápido e fila de revisão
                  |
                  v
Banco SQLite/Postgres + API somente leitura
  JSON, metadados, caminho local e URL HTTP da fotografia
                  |
       +----------+-----------+
       |                      |
Consultor Web leve      Sistema de 2ª via
Chrome/WebView/Tauri     abre foto + usa os campos da API
```

## Componentes por função

| Função | Componente recomendado | Decisão |
|---|---|---|
| Captura e câmera | PySide6 + OpenCV | Manter; já integra câmera, fila e janela nativa. |
| Detecção de folha | OpenCV: contorno, `getPerspectiveTransform`, `warpPerspective` | Manter como camada rápida; não gravar transformação no original. |
| OCR rápido | Tesseract/RapidOCR | Primeira passada por registro e nome candidato. |
| Correção manuscrita | Qwen2-VL | Somente candidatos abaixo do limiar; nunca para todas as páginas. |
| Metadados | SQLite atual, depois PostgreSQL se houver rede/múltiplos usuários | Preservar histórico e origem de cada detecção. |
| API | HTTP JSON atual | Acrescentar URL, caminho, hash e tamanho da imagem. |
| Balcão | HTML/CSS/JavaScript servido pela API | Mais rápido para consulta e abre no Chrome. |
| Aplicativo instalável opcional | Tauri/WebView2 | Usar somente se os escreventes precisarem de atalho próprio. |

## Componentes que não recomendo agora

- Electron: funciona, mas empacota um navegador inteiro; é exagerado para uma
  tela que só pesquisa e mostra fotos.
- docTR no caminho crítico da captura: tem orientação, retificação e layout,
  mas adiciona modelos e latência. Pode ser laboratório/QA, não o disparo da
  câmera.
- Binarização obrigatória: pode destruir a tinta azul e reduzir a leitura da
  manuscrita; manter versões derivadas, nunca substituir o bruto.

## Contrato mínimo para o sistema de segunda via

Cada item de `/api/v1/registros` deve trazer:

- `registro_id`, `livro_codigo`, `acervo_nome`, `oficio_nome`, `termo`, `folha` e
  `face`;
- `metadados`, com valor, confiança, status, motor e fonte;
- `caminho_imagem`, para integração no mesmo Windows;
- `imagem_url`/`foto_url`, para integração por HTTP;
- `imagem`, com `id`, `url`, `path`, `nome`, `mime`, `tamanho_bytes`, `sha256` e
  rotação.

O sistema de segunda via deve usar a URL quando estiver em outra máquina e o
`path` somente quando estiver no mesmo computador/rede de arquivos.

## Ordem de execução

1. Fechar captura, recaptura, duplicidade e histórico sem apagar originais.
2. Rodar OCR rápido por registro e enviar apenas `nome_incerto` ao Qwen.
3. Testar a API com o sistema de segunda via.
4. Criar o Consultor Web leve.
5. Só depois avaliar Tauri, PostgreSQL ou treinamento de HTR por livro.
