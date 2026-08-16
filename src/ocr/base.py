from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRToken:
    tipo: str
    valor: str
    confianca: float = 0.0
    motor: str = ""
    bbox: list | None = None

    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "valor": self.valor,
            "confianca": self.confianca,
            "motor": self.motor,
            "bbox": self.bbox,
        }


@dataclass
class OCRResult:
    tokens: list[OCRToken] = field(default_factory=list)
    texto_bruto: str = ""
    motor: str = ""
    tempo_ms: float = 0.0

    @property
    def termos(self) -> list[OCRToken]:
        return [t for t in self.tokens if t.tipo == "termo"]

    @property
    def folhas(self) -> list[OCRToken]:
        return [t for t in self.tokens if t.tipo == "folha"]

    @property
    def numeros(self) -> list[OCRToken]:
        return [t for t in self.tokens if t.tipo == "numero"]


class OCRProvider(ABC):
    @abstractmethod
    def recognize(self, image) -> OCRResult:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
