"""Entrada do aplicativo separado de consulta do acervo."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Settings, data_dir
from src.consulta.main_window import ConsultaMainWindow


def configurar_logging() -> None:
    pasta = data_dir() / "logs"
    pasta.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(pasta / "consulta.log", encoding="utf-8")],
    )


def main() -> None:
    configurar_logging()
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("ConsultaAcervo")
    app.setOrganizationName("CRC")
    settings = Settings()
    if not settings.path.exists():
        settings.save()
    janela = ConsultaMainWindow(settings)
    janela.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
