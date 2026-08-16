"""Entry point do Digitalizador de Livros."""

from __future__ import annotations

import sys
import os
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Settings, data_dir, APP_NAME
from src.ui.main_window import MainWindow


def configurar_logging():
    log_dir = data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "digitalizador.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)


def main():
    configurar_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando %s", APP_NAME)

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("CRC")

    settings = Settings()
    
    if not settings.path.exists():
        settings.save()

    window = MainWindow(settings)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
