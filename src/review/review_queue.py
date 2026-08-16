from __future__ import annotations

import logging
from typing import Callable

from ..database.repository import Repository

logger = logging.getLogger(__name__)


class ReviewQueue:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    @property
    def pendentes(self) -> int:
        return self.repo.contar_revisoes_pendentes()

    def listar(self) -> list[dict]:
        return self.repo.listar_revisoes_pendentes()

    def resolver(self, revisao_id: int) -> None:
        self.repo.resolver_revisao(revisao_id)

    def adicionar(self, imagem_id: int, tipo: str, detalhes: str = "") -> int:
        return self.repo.criar_revisao(
            imagem_id=imagem_id,
            tipo=tipo,
            detalhes=detalhes,
        )
