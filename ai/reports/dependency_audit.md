# Auditoria de dependências — M0 (2026-08-16)

Regra: **nada é instalado antes desta auditoria** e antes de uma task provar
a necessidade. Estado verificado por `importlib.import_module` no `.venv`.

## Candidatas avaliadas (todas AUSENTES no M0)

| Pacote | Instalada? | Necessidade | Benefício | Risco | PyInstaller | Decisão |
|---|---|---|---|---|---|---|
| `mediapipe` | Não (experimento em M3, desinstalado) | M3: mão/oclusão na captura | Landmarks de mão robustos vs heurística atual | Benchmark M3-T02: 1/9 acertos nos cenários (blocos de pele não são mãos anatômicas), 24,7 ms/frame vs 8,2 ms da heurística; PyInstaller complexo | Médio (binários) | **Rejeitado em M3** — heurística 8/9 vence no conjunto; reavaliar apenas com frames reais de mãos (dedos finos) |
| `pytest-qt` | Não | M6: testar widgets PySide6 | Asserts de UI com event loop | Mais setup de teste; offscreen já cobre muito | n/d (dev) | **Adiar** até M6 (UX) |
| `pytest-timeout` | Não | M1: deadlock em worker/fila | Timeout automático por teste | Baixo (dev-only) | n/d (dev) | **Adiar** para M1 (será a primeira a entrar, se M1 aprovar) |
| `ruff` | Não | Lint consistente | Padronização rápida | Baixo (dev-only) | n/d (dev) | **Adiar** — opcional, sem urgência |
| `psutil` | Não | M1: métricas de CPU/RAM do Qwen | Telemetria de recursos | Baixo | OK | **Adiar** para M1 (telemetria) |

## O que JÁ está instalado e é usado
PySide6, opencv (5.0.0.93), torch, transformers, tesseract, rapidocr_onnxruntime,
PyMuPDF, PyYAML, imagehash, rapidfuzz, scikit-image, huggingface_hub, pytest 9.1.1,
numpy/pandas etc.

## Conclusão M0
- **Nenhuma instalação neste milestone.**
- Pendências para milestones futuros: pytest-timeout/psutil em M1 (após nova
  auditoria pontual), mediapipe em M3 (com benchmark), pytest-qt em M6.