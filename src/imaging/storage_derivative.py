"""Copia de armazenamento com DPI controlado, sem substituir o original.

O arquivo de captura continua sendo a fonte imutavel. Esta rotina cria uma
derivada JPEG para armazenamento quando a fonte tem mais resolucao fisica que
o alvo, preservando cor, EXIF/ICC quando disponiveis e usando Lanczos para
reduzir a imagem sem threshold, nitidez agressiva ou perda de manuscrito.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class StorageDerivativeResult:
    source_path: str
    output_path: str
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    source_dpi: tuple[float, float] | None
    output_dpi: tuple[float, float]
    scale: float
    source_bytes: int
    output_bytes: int
    output_sha256: str
    quality: int
    status: str

    @property
    def reduction_percent(self) -> float:
        if not self.source_bytes:
            return 0.0
        return (1.0 - self.output_bytes / self.source_bytes) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_size": list(self.source_size),
            "output_size": list(self.output_size),
            "source_dpi": list(self.source_dpi) if self.source_dpi else None,
            "output_dpi": list(self.output_dpi),
            "scale": round(self.scale, 6),
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "reduction_percent": round(self.reduction_percent, 2),
            "quality": self.quality,
            "status": self.status,
        }


def _normalizar_dpi(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = (value, value)
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if x <= 0 or y <= 0:
        return None
    return x, y


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tamanho_alvo(
    size: tuple[int, int],
    source_dpi: tuple[float, float] | None,
    target_dpi: float,
) -> tuple[tuple[int, int], float, str, tuple[float, float]]:
    if target_dpi <= 0:
        raise ValueError("target_dpi deve ser positivo")
    if source_dpi is None:
        # Sem DPI de origem, nao inventamos uma escala fisica. A imagem fica
        # intacta em pixels e recebe uma copia com o DPI solicitado, deixando
        # a ausencia da medicao explicita no status.
        return size, 1.0, "dpi_origem_ausente_sem_redimensionar", (target_dpi, target_dpi)

    sx = min(1.0, target_dpi / source_dpi[0])
    sy = min(1.0, target_dpi / source_dpi[1])
    output = (
        max(1, round(size[0] * sx)),
        max(1, round(size[1] * sy)),
    )
    if sx == 1.0 and sy == 1.0:
        status = "ja_nao_acima_do_alvo"
    elif abs(sx - sy) > 1e-6:
        raise ValueError("DPI nao uniforme; redimensionamento anisotropico recusado")
    else:
        status = "redimensionada"
    return output, min(sx, sy), status, (target_dpi, target_dpi)


def criar_derivada_armazenamento(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_dpi: float = 300,
    jpeg_quality: int = 75,
    jpeg_subsampling: int = 2,
) -> StorageDerivativeResult:
    """Cria uma copia para armazenamento e nunca sobrescreve a fonte.

    A dimensao e calculada pela relacao DPI-origem/DPI-alvo. Se a origem nao
    possuir DPI, nao reduzimos pixels silenciosamente; apenas registramos o
    caso no resultado para revisao da origem.
    """
    source = Path(source_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        raise ValueError("A derivada precisa ter caminho diferente do original")
    quality = max(1, min(100, int(jpeg_quality)))
    subsampling = int(jpeg_subsampling)
    if subsampling not in (0, 1, 2):
        raise ValueError("jpeg_subsampling deve ser 0, 1 ou 2")

    with Image.open(source) as original:
        source_size = tuple(int(value) for value in original.size)
        source_dpi = _normalizar_dpi(original.info.get("dpi"))
        output_size, scale, status, output_dpi = _tamanho_alvo(
            source_size, source_dpi, float(target_dpi)
        )
        image = original
        if output_size != source_size:
            image = original.resize(output_size, Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_kwargs: dict[str, Any] = {
            "format": "JPEG",
            "quality": quality,
            "subsampling": subsampling,
            # A otimização Huffman é muito lenta em páginas grandes e não
            # melhora a fidelidade visual. O ganho de armazenamento vem do
            # redimensionamento; manter a gravação direta deixa a fila
            # previsível para lotes extensos.
            "optimize": False,
            "dpi": output_dpi,
        }
        if original.info.get("icc_profile"):
            save_kwargs["icc_profile"] = original.info["icc_profile"]
        if original.info.get("exif"):
            save_kwargs["exif"] = original.info["exif"]
        image.save(temporary, **save_kwargs)
        temporary.replace(destination)

    return StorageDerivativeResult(
        source_path=str(source),
        output_path=str(destination),
        source_size=source_size,
        output_size=output_size,
        source_dpi=source_dpi,
        output_dpi=output_dpi,
        scale=scale,
        source_bytes=source.stat().st_size,
        output_bytes=destination.stat().st_size,
        output_sha256=_sha256(destination),
        quality=quality,
        status=status,
    )
