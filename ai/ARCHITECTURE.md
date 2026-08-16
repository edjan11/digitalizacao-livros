# ARCHITECTURE — Mapa real do sistema

> Levantado no M0 (2026-08-16) a partir do código-fonte. Caminhos relativos a `D:\T.I\digitalizacao-livros`.

## Entrypoints
| Arquivo | Papel |
|---|---|
| `app.py` | Digitalizador (PySide6: câmera, captura, revisão) |
| `consulta.py` | ConsultaAcervo (busca + revisor + API local) |
| `DigitalizadorLivros.spec` / `ConsultaAcervo.spec` | PyInstaller |

## Camadas / módulos
| Camada | Arquivos | Conteúdo |
|---|---|---|
| Config | `src/config/settings.py` | config.yaml + segredos DPAPI; `data_dir()=C:\ProgramData\DigitalizadorLivros` |
| Captura | `src/capture/auto_capture.py` | `AutoCaptureController` (máquina de estados: posicionar→mão→enquadrar→foco→estabilidade→capturar→bloqueio "troque a página"); `pontuacao_mao` (heurística de pele, sem MediaPipe) |
| UI captura | `src/ui/scan_screen.py`, `src/ui/camera_capture_dialog.py`, `src/ui/main_window.py` | Tela de captura com workers em background e filas (`OCRWorker`, `CapturaPipelineWorker`) |
| Visão/geometria | `src/imaging/document.py` | `retificar_formulario` (Hough + warp), `detectar_quadrilatero_pagina`, `pagina_cortada_na_borda` |
| Qualidade | `src/imaging/quality.py` | foco (Laplacian), exposição, enquadramento, desvio; **sem detecção de reflexo/glare (gap)** |
| Recortes | `src/imaging/record_regions.py`, `book_layouts.py` | `bbox_registro` (0.045–0.76), faixa de nome, layouts A-16/A-07 |
| OCR rápido | `src/ocr/tesseract_engine.py`, `rapidocr_engine.py`, `engines.py`, `combiner.py`, `name_candidates.py` | Tesseract + RapidOCR; extração de termo/folha; localizador da linha "que recebeu o nome de" |
| IA manuscrito | `src/ocr/qwen_vl_engine.py` | Qwen2-VL-2B (CPU, float32, 24 threads); `QwenAreaAnalyzer`/`QwenRecordAnalyzer`; `preparar_imagem_qwen` (CLAHE) |
| IA descartada | `src/ocr/got_ocr_engine.py`, `htr_engine.py` | GOT-OCR2 (mantido p/ indexação secundária), HTR |
| Fila persistente | `src/services/name_processing.py` | `NameBatchRunner`: lock exclusivo, `preparar_retomada_lote`, OCR rápido paralelo (4), Qwen sequencial, tentativas (2), status `pausado/concluido/falhou` |
| Pipeline captura | `src/services/scan_pipeline.py` | `processar_imagem_imediato`, `processar_ocr_secundario`, duplicidade (SSIM), termo/folha |
| Importação | `src/services/organized_book_importer.py`, `generic_book_importer.py` | A-07 (auditoria visual, 1194 assentos), A-16 |
| Banco | `src/database/connection.py`, `repository.py` | SQLite; tabelas imagem/registro/ocr_execucao/ocr_deteccao/processamento_*/livro/acervo/oficio/tipo |
| API | `src/services/acervo_api.py` | HTTP JSON somente leitura `127.0.0.1:8765/api/v1` |
| Consulta | `src/consulta/main_window.py`, `src/ui/image_reader_window.py`, `image_viewer.py` | Busca, revisor, moldura=recorte, evidência de termo, retificação na exibição |
| Storage | `src/storage/`, `src/session/`, `src/watcher/` | Derivados, sessão de captura, watcher de pasta |
| Testes | `tests/` (13 arquivos, 77 testes) | ver `ai/TEST_MATRIX.md` |

## Fluxo principal (captura → busca)
```
câmera → AutoCaptureController (prévia)
   → scan_pipeline.processar_imagem_imediato (hash, qualidade, duplicidade) [background]
   → processar_ocr_secundario (OCR rápido por registro) [background, fila]
   → linha "que recebeu o nome de" → sugestão de nome
   → incertos → fila Qwen persistente (nome+termo+mãe+data)
   → revisor: confirmar/corrigir (única forma de virar verde)
   → API read-only / ConsultaAcervo (busca + foto)
```

## Fluxo de dados de leitura (rastreabilidade)
```
ocr_execucao (motor, texto_bruto, tempo, ativo)
   └─ ocr_deteccao (tipo, valor_original, valor_tratado, valor_normalizado,
                    confianca, motor, fonte, status, bbox_json, contexto)
```

## Gap registrados no M0
- Reflexo/glare: inexistente em `quality.py`.
- Telemetria por etapa (tempo/fila/taxa): inexistente.
- Estados de captura implícitos (strings) em vez de enum formal.
- Painel de recaptura por folha com motivo: inexistente.
- Busca por livro estilo Explorer (2 painéis): inexistente (só filtros).
- FTS5: não avaliado.