# Baseline — M0 (2026-08-16)

## Testes
```
77 passed in 363.62s (0:06:03)
77 tests collected in 1.93s
```
- Ambiente: Python 3.11.3 (venv `.venv`), 24 cores, CPU-only (sem GPU NVIDIA).
- PySide6 6.11.1 · OpenCV 5.0.0.93 · torch 2.13.0 · transformers 5.14.1
  · rapidocr_onnxruntime 1.4.4 · pytest 9.1.1 · imagehash 4.3.2 · rapidfuzz 3.14.5

## Git
- Repositório criado no M0: `master`, commit raiz `1b7266d`
  "baseline: estado atual do Digitalizador/ConsultaAcervo (pre-loop)" (121 arquivos, 20.781 linhas).
- `.gitignore` exclui: `.venv*`, `build*`, `dist*`, `__pycache__`, `.data/`,
  `.tmp_*`, `models/`, `*.log`, `.pytest_cache`, specs antigos.

## Inventário (resumo)
- 2 entrypoints: `app.py` (Digitalizador) e `consulta.py` (ConsultaAcervo).
- Captura: `src/capture/auto_capture.py` (estados já implementados) + `src/ui/scan_screen.py` (workers em background).
- Fila persistente: `src/services/name_processing.py` (lock exclusivo + retomada) — validada por testes.
- OCR: Tesseract + RapidOCR; IA: Qwen2-VL-2B local; SQLite; API read-only 8765; PyInstaller ok.

## Divergências vs. premissas do plano original
| Premissa do plano | Realidade |
|---|---|
| Projeto tem git | Não tinha → git criado no M0 |
| M2 (capture state machine) a construir | Já existe ~85% (`AutoCaptureController` + 6 testes) |
| Job em background pós-captura a criar | Já existe (`OCRWorker`/`CapturaPipelineWorker` + filas) |
| Mão a detectar | Já existe heurística (`pontuacao_mao`); MediaPipe ausente |
| Glare a detectar | **Não existe** — gap real para M3 |
| Telemetria | **Não existe** — gap para M1 |
| Recaptura por folha | Não existe — gap para M5 |
| Busca Explorer | Não existe — gap para M7 |

## Riscos abertos
1. Qwen lento (16,5–48 s/leitura) em CPU — mitigável com GPU/nuvem (fora de escopo).
2. Reflexo/glare pode gerar fotos ruins silenciosamente — M3/M4.
3. Operador não tem visão do que falta no livro — M4/M5.
4. Duas instâncias da Consulta abertas simultaneamente disputam porta/API — orientar no release.

## Recomendações para M1
1. Telemetria por etapa em JSONL (tempo, fila, taxa de incertos, motores).
2. Testes de caracterização sob carga da fila (10× itens, 2 workers) e de
   transição de estados da captura (matriz de transições).
3. Adicionar `pytest-timeout` + `psutil` (auditados) para detectar deadlock e medir recursos.
4. Primeiras tasks M1: `M1-T01` telemetria do worker de nomes; `M1-T02` telemetria do pipeline de captura;
   `M1-T03` matriz de transições da captura; `M1-T04` teste de carga da fila.