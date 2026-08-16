"""Engine Tesseract OCR via subprocess (binario embutido no pacote)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


class TesseractError(Exception):
    pass


class TesseractEngine:
    def __init__(self, tesseract_path: str | None = None, tessdata_path: str | None = None, lang: str = "por") -> None:
        self.tesseract_path = tesseract_path or self._locate()
        self.tessdata_path = tessdata_path or self._locate_tessdata()
        self.lang = self._resolve_lang(lang)

    @staticmethod
    def _locate() -> str:
        exe = shutil.which("tesseract")
        if exe:
            return exe
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if Path(candidate).exists():
                return candidate
        frozen = Path(getattr(__import__("sys"), "executable", "")).parent / "tesseract" / "tesseract.exe"
        if frozen.exists():
            return str(frozen)
        return ""

    def _locate_tessdata(self) -> str:
        """Busca tessdata em multiplas pastas, priorizando a do app."""
        candidates = []
        if self.tesseract_path:
            candidates.append(str(Path(self.tesseract_path).parent / "tessdata"))
        # Pasta do app (ProgramData)
        app_data = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "OrganizadorFirmas" / "tessdata"
        candidates.append(str(app_data))
        # Pasta do frozen
        frozen = Path(getattr(__import__("sys"), "executable", "")).parent / "tesseract" / "tessdata"
        candidates.append(str(frozen))

        for path in candidates:
            if Path(path).exists():
                return path
        return ""

    def _resolve_lang(self, lang: str) -> str:
        """Verifica se o idioma existe; se nao, tenta fallback para eng."""
        if not self.tessdata_path:
            return lang
        tessdata = Path(self.tessdata_path)
        target = tessdata / f"{lang}.traineddata"
        if target.exists():
            return lang
        eng = tessdata / "eng.traineddata"
        if eng.exists():
            return "eng"
        return lang

    def is_available(self) -> bool:
        return bool(self.tesseract_path) and Path(self.tesseract_path).exists()

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.tessdata_path:
            env["TESSDATA_PREFIX"] = self.tessdata_path
        elif self.tesseract_path:
            env["TESSDATA_PREFIX"] = str(Path(self.tesseract_path).parent / "tessdata")
        return env

    def _user_words_path(self) -> Path | None:
        if self.tessdata_path:
            p = Path(self.tessdata_path) / "user-words.txt"
            if p.exists():
                return p
        return None

    def read(self, image_path: Path, psm: int = 6) -> str:
        """Executa o Tesseract sobre um recorte e devolve o texto bruto."""
        if not self.is_available():
            raise TesseractError("Tesseract nao localizado")

        cmd = [self.tesseract_path, str(image_path), "stdout", "--psm", str(psm), "-l", self.lang]
        uw = self._user_words_path()
        if uw:
            cmd.extend(["--user-words", str(uw)])
        import platform
        kwargs = dict(
            capture_output=True,
            text=True,
            timeout=60,
            env=self._env(),
            encoding="utf-8",
            errors="replace",
        )
        if platform.system() == "Windows":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        try:
            result = subprocess.run(
                cmd,
                **kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            raise TesseractError("Tesseract excedeu o tempo limite") from exc
        except FileNotFoundError as exc:
            raise TesseractError("Binario do Tesseract nao encontrado") from exc

        if result.returncode != 0:
            raise TesseractError(f"Tesseract falhou (rc={result.returncode}): {result.stderr[-300:]}")
        return result.stdout

    def read_array(self, image: "cv2.Mat | None", psm: int = 6) -> str:
        """OCR sobre buffer numpy via arquivo temporario."""
        if image is None:
            return ""
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), image)
            return self.read(tmp_path, psm=psm)
        finally:
            tmp_path.unlink(missing_ok=True)

    def detectar_orientacao(self, image: "cv2.Mat | None") -> dict:
        """Detecta a orientacao do texto via OSD (--psm 0).

        Retorna dict com 'rotacao' (0, 90, 180, 270) e 'confianca'.
        Uma pagina digitalizada de cabeca para baixo retorna 180.
        """
        if image is None:
            return {"rotacao": 0, "confianca": 0.0}
        if not self.is_available():
            return {"rotacao": 0, "confianca": 0.0}
        if not Path(self.tessdata_path or "").joinpath("osd.traineddata").exists():
            return {"rotacao": 0, "confianca": 0.0}

        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), image)
            cmd = [self.tesseract_path, str(tmp_path), "stdout", "--psm", "0"]
            import platform
            kwargs = dict(
                capture_output=True,
                text=True,
                timeout=60,
                env=self._env(),
                encoding="utf-8",
                errors="replace",
            )
            if platform.system() == "Windows":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            result = subprocess.run(cmd, **kwargs)
            if result.returncode != 0:
                return {"rotacao": 0, "confianca": 0.0}
            texto = result.stdout
        finally:
            tmp_path.unlink(missing_ok=True)

        rotacao = 0
        confianca = 0.0
        for linha in texto.splitlines():
            if linha.startswith("Rotate:"):
                try:
                    rotacao = int(linha.split(":")[1].strip())
                except ValueError:
                    rotacao = 0
            elif "Orientation confidence:" in linha:
                try:
                    confianca = float(linha.split(":")[1].strip())
                except ValueError:
                    confianca = 0.0
        return {"rotacao": rotacao, "confianca": confianca}