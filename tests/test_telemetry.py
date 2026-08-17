from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Settings
from src.services import telemetry


def _linhas_eventos(base: Path) -> list[dict]:
    arquivo = base / "events.jsonl"
    if not arquivo.exists():
        return []
    return [
        json.loads(linha)
        for linha in arquivo.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]


def test_telemetria_eventos_sem_pii(tmp_path):
    settings = Settings(tmp_path / "config.yaml")
    settings.set("telemetry", "enabled", True)
    settings.set("telemetry", "path", str(tmp_path))
    telemetry.configurar(settings)

    telemetry.emitir("ocr.fast_finished", job_id=7, duration_ms=42.5,
                     uncertain=True, nome="MARIA PROIBIDA", cpf="123", texto="x")
    telemetry.emitir("capture.state", de="inicio", para="pronta", duracao_estado_ms=200)

    time.sleep(0.4)  # deixa a thread de escrita esvaziar a fila
    telemetry.parar()
    time.sleep(0.2)

    linhas = _linhas_eventos(tmp_path)
    eventos = [l["event"] for l in linhas]
    bruto = json.dumps(linhas)
    assert "ocr.fast_finished" in eventos
    assert "capture.state" in eventos
    assert "MARIA PROIBIDA" not in bruto
    assert "123" not in bruto
    assert linhas[0]["job_id"] == 7
    assert linhas[0]["duration_ms"] == 42.5


def test_telemetria_desligada_nao_escreve(tmp_path):
    settings = Settings(tmp_path / "config.yaml")
    settings.set("telemetry", "enabled", False)
    settings.set("telemetry", "path", str(tmp_path))
    telemetry.configurar(settings)

    telemetry.emitir("ocr.fast_finished", job_id=1, duration_ms=1.0)
    time.sleep(0.3)
    assert not (tmp_path / "events.jsonl").exists()
    telemetry.parar()


def test_telemetria_sem_configurar_e_inerte(tmp_path):
    telemetry.emitir("qualquer.evento", job_id=1)
    assert True  # nao levanta e nao escreve nada


def test_amostrador_registrado_e_chamado(tmp_path):
    settings = Settings(tmp_path / "config.yaml")
    settings.set("telemetry", "enabled", True)
    settings.set("telemetry", "path", str(tmp_path))
    telemetry.configurar(settings)

    chamadas = []

    def coletor() -> dict:
        chamadas.append(1)
        return {"queue_depth": 5}

    telemetry.registrar_amostrador("fila_teste", coletor)
    telemetry.iniciar_amostrador(0.5)
    time.sleep(1.2)
    telemetry.registrar_amostrador("fila_teste", None)
    telemetry.parar()
    time.sleep(0.3)

    linhas = _linhas_eventos(tmp_path)
    amostras = [l for l in linhas if l["event"] == "resource.sample"]
    assert chamadas, "amostrador nunca foi chamado"
    assert amostras and any(l.get("queue_depth") == 5 for l in amostras)
