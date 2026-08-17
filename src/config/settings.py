from __future__ import annotations

import ctypes
import copy
import os
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

APP_NAME = "DigitalizadorLivros"
APP_VERSION = "1.1.0"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    override = os.environ.get("DIGITALIZADOR_DATA_DIR", "").strip()
    if override:
        d = Path(override)
    else:
        # Fonte e executaveis precisam enxergar o mesmo acervo operacional.
        # Bancos temporarios de testes continuam sendo passados diretamente a
        # Database ou pelo override acima; nunca mesclamos .data implicitamente.
        base = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
        d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def resource_dir() -> Path:
    if is_frozen():
        return app_root()
    return app_root() / "packaging"


def default_config_path() -> Path:
    return data_dir() / "config.yaml"


class SettingsError(Exception):
    pass


class Settings:
    DEFAULTS: dict[str, Any] = {
        "czur": {
            "watch_folder": r"D:\CZUR\Scans",
            "debounce_ms": 500,
        },
        "camera": {
            "index": 0,
        },
        "acervo": {
            "root_path": r"D:\AcervoLivros",
        },
        "imaging": {
            "dpi": 300,
            "jpeg_quality": 90,
            "storage_dpi": 300,
            "storage_jpeg_quality": 75,
            "thumb_width": 200,
        },
        "ocr": {
            "tesseract_path": "",
            "tessdata_path": "",
            "lang": "por",
            "rapidocr_enabled": True,
            "tesseract_enabled": True,
            # O EasyOCR e util no laboratorio, mas e lento e nao e um HTR
            # treinado para estes manuscritos. A captura usa a sequencia do
            # livro e OCR rapido; habilite apenas para testes comparativos.
            "htr_enabled": False,
            # GOT-OCR 2.0 é opcional e roda somente na indexação da Consulta.
            # Na primeira utilização os pesos (~1,1 GB) são baixados para a
            # pasta de dados do aplicativo; nesta estação a inferência usa CPU.
            "got_model_path": "",
            "got_max_new_tokens": 384,
            # Qwen2-VL só é usado na leitura pontual de uma área selecionada.
            "qwen_model_path": "",
            "qwen_max_new_tokens": 96,
            "qwen_min_pixels": 128 * 28 * 28,
            "qwen_max_pixels": 384 * 28 * 28,
            "qwen_dtype": "auto",
            "qwen_threads": 0,
            # O OCR tradicional levou 6--10 s e quase sempre terminou no
            # Qwen nestes dois livros manuscritos. Encaminhar a faixa
            # calibrada diretamente evita esse custo redundante.
            "name_direct_qwen_books": ["A-07", "A-16"],
            # O manuscrito A-07 foi mais fiel sem CLAHE. Ative somente para
            # comparar uma cópia de contraste, nunca para substituir o bruto.
            "qwen_preprocess": False,
            # Abaixo deste valor o nome fica marcado para correção pontual no
            # Qwen; nomes prováveis permanecem apenas como sugestão pesquisável.
            "name_qwen_threshold": 0.78,
            "agree_threshold": 0.7,
        },
        "quality": {
            "blur_threshold": 100,
            "dark_threshold": 40,
            "overexposed_pct": 30,
            "skew_max_degrees": 3.0,
        },
        "duplicate": {
            "phash_threshold": 5,
            "dhash_threshold": 8,
            "ssim_threshold": 0.95,
            "ssim_suspect": 0.90,
        },
        "ui": {
            "language": "pt-BR",
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8765,
        },
        "telemetry": {
            # Eventos de desempenho em JSONL (sem imagem, sem texto, sem PII).
            # Desligar aqui nao altera nenhuma funcionalidade.
            "enabled": True,
            "path": "",
        },
    }

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_config_path()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                raise SettingsError(f"Nao foi possivel ler config: {exc}") from exc
        else:
            loaded = {}
        # As seções são dicionários aninhados. Uma cópia rasa fazia um teste
        # ou uma alteração temporária contaminar os padrões das próximas
        # instâncias de Settings no mesmo processo.
        self._data = self._deep_merge(copy.deepcopy(self.DEFAULTS), loaded)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Settings._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(yaml.safe_dump(self._data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        self._data.setdefault(section, {})[key] = value

    def section(self, name: str) -> dict:
        return dict(self._data.get(name, {}))
