from __future__ import annotations

import logging

from ..database.repository import Repository

logger = logging.getLogger(__name__)


class ScanSession:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.oficio_id: int | None = None
        self.tipo_id: int | None = None
        self.subtipo_id: int | None = None
        self.livro_id: int | None = None
        self.livro: dict | None = None
        self.ultima_folha: int | None = None
        self.ultima_face: str = "frente"
        self.ultimo_termo: int | None = None
        self.pasta_monitorada: str | None = None
        self._carregar()

    def _carregar(self) -> None:
        saved = self.repo.carregar_sessao()
        if saved:
            self.oficio_id = saved.get("oficio_id")
            self.tipo_id = saved.get("tipo_id")
            self.subtipo_id = saved.get("subtipo_id")
            self.livro_id = saved.get("livro_id")
            self.pasta_monitorada = saved.get("pasta_monitorada")
            if self.livro_id:
                self.livro = self.repo.get_livro(self.livro_id)
            self.ultima_folha = saved.get("ultima_folha")
            self.ultima_face = saved.get("ultima_face") or "frente"
            self.ultimo_termo = saved.get("ultimo_termo")

    def tem_sessao_ativa(self) -> bool:
        return self.livro_id is not None and self.livro is not None

    def selecionar_oficio(self, oficio_id: int) -> None:
        self.oficio_id = oficio_id
        self._salvar()

    def selecionar_tipo(self, tipo_id: int) -> None:
        self.tipo_id = tipo_id
        self.subtipo_id = None
        self._salvar()

    def selecionar_subtipo(self, subtipo_id: int) -> None:
        self.subtipo_id = subtipo_id
        self._salvar()

    def selecionar_livro(self, livro_id: int) -> None:
        self.livro_id = livro_id
        self.livro = self.repo.get_livro(livro_id)
        ultima = self.repo.get_ultima_imagem_nao_duplicada(livro_id)
        if ultima:
            self.ultima_folha = ultima.get("folha_estimada")
            self.ultima_face = ultima.get("face") or "frente"
            self.ultimo_termo = ultima.get("termo_inicial")
            self._avancar_estado()
        else:
            self.ultima_folha = self.livro.get("primeira_folha") if self.livro else None
            self.ultima_face = "frente"
            self.ultimo_termo = self.livro.get("termo_inicial") if self.livro else None
        self._salvar()

    def avancar_pagina(
        self,
        termo_encontrado: int | None = None,
        registros_na_face: int | None = None,
    ) -> None:
        self._avancar_estado(termo_encontrado, registros_na_face)
        self._salvar()

    def _avancar_estado(
        self,
        termo_encontrado: int | None = None,
        registros_na_face: int | None = None,
    ) -> None:
        if self.ultima_face == "frente" and self.livro and self.livro.get("frente_verso"):
            self.ultima_face = "verso"
        else:
            self.ultima_face = "frente"
            if self.ultima_folha is not None:
                self.ultima_folha += 1
        if termo_encontrado is not None:
            self.ultimo_termo = termo_encontrado
        elif self.ultimo_termo is not None and self.livro:
            reg_por_face = registros_na_face or self.livro.get("registros_por_face", 1)
            self.ultimo_termo += reg_por_face

    def intervalo_termos_com(self, registros_na_face: int | None = None) -> tuple[int | None, int | None]:
        """Retorna a faixa da face usando a quantidade detectada, quando houver."""
        if self.ultimo_termo is None:
            return None, None
        quantidade = max(
            1,
            int(registros_na_face or (self.livro or {}).get("registros_por_face", 1)),
        )
        final = self.ultimo_termo + quantidade - 1
        if self.livro and self.livro.get("termo_final") is not None:
            final = min(final, int(self.livro["termo_final"]))
        return self.ultimo_termo, final

    def limpar_sessao(self) -> None:
        self.oficio_id = None
        self.tipo_id = None
        self.subtipo_id = None
        self.livro_id = None
        self.livro = None
        self.ultima_folha = None
        self.ultima_face = "frente"
        self.ultimo_termo = None
        self.repo.salvar_sessao(
            oficio_id=None, tipo_id=None, subtipo_id=None,
            livro_id=None, ultima_folha=None, ultima_face=None,
            ultimo_termo=None, pasta_monitorada=None,
        )

    def _salvar(self) -> None:
        self.repo.salvar_sessao(
            oficio_id=self.oficio_id, tipo_id=self.tipo_id,
            subtipo_id=self.subtipo_id, livro_id=self.livro_id,
            ultima_folha=self.ultima_folha, ultima_face=self.ultima_face,
            ultimo_termo=self.ultimo_termo, pasta_monitorada=self.pasta_monitorada,
        )

    @property
    def proximo_termo_esperado(self) -> int | None:
        if self.ultimo_termo is None:
            return None
        regs = self.livro.get("registros_por_face", 1) if self.livro else 1
        return self.ultimo_termo + regs

    @property
    def intervalo_termos_atual(self) -> tuple[int | None, int | None]:
        """Intervalo inclusivo esperado para a proxima imagem.

        ``ultimo_termo`` representa o primeiro termo da proxima face. Uma face
        com dois registros iniciada no termo 6801, portanto, contem 6801-6802;
        6803 e o inicio da face seguinte.
        """
        if self.ultimo_termo is None:
            return None, None
        regs = max(1, int(self.livro.get("registros_por_face", 1))) if self.livro else 1
        final = self.ultimo_termo + regs - 1
        if self.livro and self.livro.get("termo_final") is not None:
            final = min(final, int(self.livro["termo_final"]))
        return self.ultimo_termo, final

    @property
    def resumo(self) -> str:
        if not self.tem_sessao_ativa():
            return "Nenhuma sessao ativa"
        oficio = self.repo.get_oficio(self.oficio_id) if self.oficio_id else None
        tipo = self.repo.get_tipo(self.tipo_id) if self.tipo_id else None
        nome_oficio = oficio.get("nome", "") if oficio else ""
        nome_tipo = tipo.get("nome", "") if tipo else ""
        livro_codigo = self.livro.get("codigo", "") if self.livro else ""
        return f"{nome_oficio} -> {nome_tipo} -> {livro_codigo}"
