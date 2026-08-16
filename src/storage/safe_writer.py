"""Gravacao segura de JPG com validacao pos-escrita."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import cv2

from ..imaging.jpeg_otimizado import codificar_jpeg, gravar_jpeg


class WriteError(Exception):
    pass


class SafeWriter:
    def __init__(self, verify_sha256: bool = True) -> None:
        self.verify = verify_sha256

    def gravar(self, dest: Path, source: Path | None = None, image_array=None) -> dict:
        """Grava arquivo com escrita atomica (temp + rename) e validacao.

        Retorna dict com sha256, dimensoes, bytes.
        """
        sha256 = ""
        width = height = 0

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=str(dest.parent)) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if source and source.exists():
                shutil.copy2(source, tmp_path)
                if image_array is not None:
                    # Re-codifica com a qualidade otimizada do sistema
                    gravar_jpeg(tmp_path, image_array)
            elif image_array is not None:
                tmp_path.write_bytes(codificar_jpeg(image_array))
            else:
                raise WriteError("Nenhuma fonte de dados para gravar")

            # Validacao
            img = cv2.imread(str(tmp_path))
            if img is None:
                raise WriteError(f"Arquivo gerado invalido: {tmp_path.name}")
            height, width = img.shape[:2]
            sha256 = hashlib.sha256(tmp_path.read_bytes()).hexdigest()

            # Rename atomico
            tmp_path.rename(dest)

            # Verificacao pos-rename
            if self.verify:
                if not dest.exists():
                    raise WriteError("Arquivo nao encontrado apos gravacao")
                file_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
                if file_sha != sha256:
                    raise WriteError("SHA-256 diverge apos gravacao")

        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise WriteError(str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "sha256": sha256,
            "width": width,
            "height": height,
            "bytes": dest.stat().st_size if dest.exists() else 0,
        }