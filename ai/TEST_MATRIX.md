# TEST_MATRIX — Mapa dos 77 testes + scripts de benchmark

> Coletado em 2026-08-16: `pytest --collect-only` → 77 testes em 13 arquivos.

## Suite (pytest)
| Arquivo | Testes | Cobre |
|---|---|---|
| `tests/test_basico.py` | 10 | Sanidade geral, helpers, extração |
| `tests/test_captura_automatica.py` | 6 | `AutoCaptureController`: estados, mão, foco, transição |
| `tests/test_consulta.py` | 10 | Metadados, busca, `_acervo_com_imagem` |
| `tests/test_processamento_persistente.py` | 16 | Fila/lote/retomada/lock/Qwen/geometria/recorte |
| `tests/test_modelos_visuais.py` | 10 | Viewer (moldura=recorte), Qwen parser, GOT token |
| `tests/test_sequencia_ocr.py` | 7 | Sequência de termo/folha, OCR |
| `tests/test_adaptive_layout.py` | 4 | Layouts de livro A-16/A-07 |
| `tests/test_page_orientation.py` | 4 | Rotação 0/90/180/270 |
| `tests/test_generic_book_import.py` | 3 | Importação genérica |
| `tests/test_storage_derivative.py` | 3 | Derivada 300 DPI (prioridade do leitor) |
| `tests/test_oriented_copy.py` | 2 | Cópia orientada p/ Chrome |
| `tests/test_acervo_api.py` | 1 | API read-only |
| `tests/test_nome_candidatos.py` | 1 | Candidatos de nome |

**Total: 77** · tempo 363.62s · sem GPU, sem modelos (mocks).

## Testes reais com foto (skip condicional)
- `test_foto_real_6801_6802_*` e `test_localizador_real_*` usam
  `D:\A - 07\FRENTE|VERSO\IMG_*.jpg` — pulados se o arquivo não existir.

## Scripts de benchmark/validação (reprodutíveis)
| Script | Mede |
|---|---|
| `scripts/benchmark_qwen_20.py` | Qwen2-VL-2B em 20 nomes (acerto+tempo) |
| `scripts/benchmark_kraken_20.py` | Kraken TraPrInq |
| `scripts/benchmark_got_20.py` | GOT-OCR 2.0 |
| `scripts/benchmark_ollama_20.py` | Qwen2.5-VL-3B via Ollama |
| `scripts/medir_pixels_qwen.py` / `medir_assento_qwen.py` | Curva pixels×tempo |
| `scripts/varrer_retificacao_livro.py` | Retificação em todas as 637 fotos do A-07 |

## Gatilhos de regressão
- Qualquer mudança em `record_regions`/`book_layouts` → rodar
  `test_processamento_persistente` + `test_modelos_visuais`.
- Qualquer mudança em `scan_pipeline` → `test_captura_automatica` + `test_consulta` + `test_sequencia_ocr`.
- Qualquer mudança em `name_processing` → `test_processamento_persistente`.
- Qualquer mudança de UI → suite offscreen + smoke do `.exe` (M8).