from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watcher_component import arquivo_estavel

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class CzurHandler(FileSystemEventHandler):
    def __init__(self, callback, debounce_ms: int = 500) -> None:
        super().__init__()
        self.callback = callback
        self.debounce_ms = debounce_ms
        self._recent: dict[str, float] = {}

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = event.src_path
        image_path = Path(path)
        ext = image_path.suffix.lower()
        if ext not in IMAGE_EXTS:
            return
        nome = image_path.name.lower()
        if "_prep" in nome or "_thumb" in nome or image_path.parent.name == ".ocr_cache":
            return
        now = time.time()
        if path in self._recent and (now - self._recent[path]) < 5:
            return
        self._recent[path] = now
        if not self._aguardar_arquivo_estavel(image_path):
            logger.warning("Imagem nao ficou pronta a tempo: %s", image_path.name)
            return
        logger.info("Nova imagem detectada: %s", Path(path).name)
        self.callback(path)

    def _aguardar_arquivo_estavel(self, path: Path) -> bool:
        """Evita abrir a foto enquanto o scanner ainda esta gravando o JPEG."""
        intervalo = max(0.05, min(self.debounce_ms / 1000.0, 0.5))
        return arquivo_estavel(
            path, timeout_seg=5.0, intervalo_seg=intervalo, leituras_iguais=2,
        )


class FolderWatcher(QObject):
    imagem_detectada = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._observer: Observer | None = None
        self._handler: CzurHandler | None = None
        self._pasta: str | None = None

    def iniciar(self, pasta: str, debounce_ms: int = 500) -> None:
        if self._observer and self._observer.is_alive():
            self.parar()
        self._pasta = pasta
        path = Path(pasta)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Pasta monitorada criada: %s", path)
        self._handler = CzurHandler(
            callback=lambda p: self.imagem_detectada.emit(p),
            debounce_ms=debounce_ms,
        )
        self._observer = Observer()
        self._observer.schedule(self._handler, str(path), recursive=False)
        self._observer.start()
        logger.info("Monitorando pasta: %s", path)

    def parar(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
            logger.info("Monitoramento parado")
