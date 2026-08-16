from __future__ import annotations

from pathlib import Path


class AcervoResolver:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def pasta_livro(self, livro_id: int) -> Path:
        return self.root / f"livro_{livro_id}"

    def pasta_original(self, livro_id: int) -> Path:
        d = self.pasta_livro(livro_id) / "original"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pasta_thumb(self, livro_id: int) -> Path:
        d = self.pasta_livro(livro_id) / "thumbs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pasta_ocr(self, livro_id: int) -> Path:
        d = self.pasta_livro(livro_id) / "ocr"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pasta_processado(self, livro_id: int) -> Path:
        d = self.pasta_livro(livro_id) / "processado"
        d.mkdir(parents=True, exist_ok=True)
        return d
