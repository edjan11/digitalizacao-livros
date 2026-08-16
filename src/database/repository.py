from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .connection import Database
from ..imaging.record_regions import bbox_corresponde_registro
from ..metadata.normalizer import normalizar_busca


def _distancia_levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(atual[-1] + 1, anterior[j] + 1,
                             anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def _nome_normalizado_fuzzy(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", texto.lower()).split())


def _similaridade_levenshtein(a: str, b: str) -> float:
    a, b = _nome_normalizado_fuzzy(a), _nome_normalizado_fuzzy(b)
    if not a or not b:
        return 0.0
    return 1.0 - _distancia_levenshtein(a, b) / max(len(a), len(b))


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def listar_oficios(self) -> list[dict]:
        return self.db.fetchall("SELECT * FROM oficio WHERE ativo=1 ORDER BY id")

    def listar_acervos(self) -> list[dict]:
        return self.db.fetchall("SELECT * FROM acervo WHERE ativo=1 ORDER BY nome")

    def get_oficio(self, oficio_id: int) -> dict | None:
        return self.db.fetchone("SELECT * FROM oficio WHERE id=?", (oficio_id,))

    def listar_tipos(self, parent_id: int | None = None) -> list[dict]:
        if parent_id is not None:
            return self.db.fetchall("SELECT * FROM tipo_registro WHERE subtipo_de=? AND ativo=1 ORDER BY id", (parent_id,))
        return self.db.fetchall("SELECT * FROM tipo_registro WHERE subtipo_de IS NULL AND ativo=1 ORDER BY id")

    def get_tipo(self, tipo_id: int) -> dict | None:
        return self.db.fetchone("SELECT * FROM tipo_registro WHERE id=?", (tipo_id,))

    def criar_livro(self, **kwargs) -> int:
        if "acervo_id" not in kwargs and kwargs.get("oficio_id") in {6, 9, 12, 13, 14, 15}:
            kwargs["acervo_id"] = kwargs["oficio_id"]
        now = kwargs.pop("created_at", None) or __import__("datetime").datetime.now().isoformat()
        cols = list(kwargs.keys()) + ["created_at", "updated_at"]
        placeholders = ", ".join("?" for _ in cols)
        vals = list(kwargs.values()) + [now, now]
        return self.db.insert(f"INSERT INTO livro ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals))

    def get_livro(self, livro_id: int) -> dict | None:
        return self.db.fetchone("SELECT * FROM livro WHERE id=?", (livro_id,))

    def listar_livros_por_categoria(self, oficio_id: int, tipo_id: int, subtipo_id: int | None = None) -> list[dict]:
        if subtipo_id is not None:
            return self.db.fetchall(
                "SELECT * FROM livro WHERE oficio_id=? AND tipo_id=? AND subtipo_id=? ORDER BY codigo",
                (oficio_id, tipo_id, subtipo_id),
            )
        return self.db.fetchall(
            "SELECT * FROM livro WHERE oficio_id=? AND tipo_id=? AND subtipo_id IS NULL ORDER BY codigo",
            (oficio_id, tipo_id),
        )

    def atualizar_livro(self, livro_id: int, **kwargs) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        kwargs["updated_at"] = now
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [livro_id]
        self.db.update(f"UPDATE livro SET {sets} WHERE id=?", tuple(vals))

    def registrar_imagem(self, **kwargs) -> int:
        now = kwargs.pop("created_at", None) or __import__("datetime").datetime.now().isoformat()
        cols = list(kwargs.keys()) + ["created_at"]
        placeholders = ", ".join("?" for _ in cols)
        vals = list(kwargs.values()) + [now]
        return self.db.insert(f"INSERT INTO imagem ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals))

    def get_imagem(self, imagem_id: int) -> dict | None:
        return self.db.fetchone("SELECT * FROM imagem WHERE id=?", (imagem_id,))

    def get_ultima_imagem_livro(self, livro_id: int) -> dict | None:
        return self.db.fetchone(
            "SELECT * FROM imagem WHERE livro_id=? ORDER BY ordem_captura DESC LIMIT 1", (livro_id,)
        )

    def get_ultima_imagem_nao_duplicada(self, livro_id: int) -> dict | None:
        return self.db.fetchone(
            """
            SELECT * FROM imagem
            WHERE livro_id=?
              AND duplicidade_status NOT IN ('duplicata_confirmada', 'possivel_duplicata')
            ORDER BY ordem_captura DESC
            LIMIT 1
            """,
            (livro_id,),
        )

    def get_imagens_livro(self, livro_id: int) -> list[dict]:
        return self.db.fetchall(
            "SELECT * FROM imagem WHERE livro_id=? ORDER BY ordem_captura", (livro_id,)
        )

    def buscar_imagem_por_termo(self, livro_id: int, termo: int) -> dict | None:
        """Localiza a face cujo intervalo inclusivo contem o termo."""
        return self.db.fetchone(
            """
            SELECT * FROM imagem
            WHERE livro_id=?
              AND termo_inicial IS NOT NULL
              AND termo_final IS NOT NULL
              AND termo_inicial <= ?
              AND termo_final >= ?
            ORDER BY ordem_captura
            LIMIT 1
            """,
            (livro_id, termo, termo),
        )

    def atualizar_imagem(self, imagem_id: int, **kwargs) -> None:
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [imagem_id]
        self.db.update(f"UPDATE imagem SET {sets} WHERE id=?", tuple(vals))

    def buscar_duplicatas(self, livro_id: int, phash: str) -> list[dict]:
        return self.db.fetchall(
            "SELECT * FROM imagem WHERE livro_id=? AND hash_perceptual=? AND id != (SELECT MAX(id) FROM imagem WHERE livro_id=?)",
            (livro_id, phash, livro_id),
        )

    def criar_ocorrencia(self, **kwargs) -> int:
        now = kwargs.pop("created_at", None) or __import__("datetime").datetime.now().isoformat()
        cols = list(kwargs.keys()) + ["created_at"]
        placeholders = ", ".join("?" for _ in cols)
        vals = list(kwargs.values()) + [now]
        return self.db.insert(f"INSERT INTO ocorrencia ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals))

    def listar_ocorrencias_livro(self, livro_id: int) -> list[dict]:
        return self.db.fetchall("SELECT * FROM ocorrencia WHERE livro_id=? ORDER BY created_at DESC", (livro_id,))

    def ocorrencia_existe(self, livro_id: int, tipo: str, folha: int | None = None, termo: int | None = None) -> bool:
        if folha is not None:
            row = self.db.fetchone(
                "SELECT id FROM ocorrencia WHERE livro_id=? AND tipo=? AND folha_afetada=? AND confirmada=1",
                (livro_id, tipo, folha),
            )
        elif termo is not None:
            row = self.db.fetchone(
                "SELECT id FROM ocorrencia WHERE livro_id=? AND tipo=? AND termo_afetado=? AND confirmada=1",
                (livro_id, tipo, termo),
            )
        else:
            row = self.db.fetchone(
                "SELECT id FROM ocorrencia WHERE livro_id=? AND tipo=? AND confirmada=1",
                (livro_id, tipo),
            )
        return row is not None

    def criar_revisao(self, **kwargs) -> int:
        now = kwargs.pop("created_at", None) or __import__("datetime").datetime.now().isoformat()
        cols = list(kwargs.keys()) + ["created_at"]
        placeholders = ", ".join("?" for _ in cols)
        vals = list(kwargs.values()) + [now]
        return self.db.insert(f"INSERT INTO revisao ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals))

    def contar_revisoes_pendentes(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM revisao WHERE resolvida=0")
        return row["cnt"] if row else 0

    def listar_revisoes_pendentes(self) -> list[dict]:
        return self.db.fetchall(
            """
            SELECT r.*,
                   i.livro_id, i.caminho_original, i.caminho_thumb,
                   i.folha_estimada, i.face, i.termo_inicial, i.termo_final,
                   i.ordem_captura
            FROM revisao r
            LEFT JOIN imagem i ON i.id=r.imagem_id
            WHERE r.resolvida=0
            ORDER BY
                CASE WHEN r.tipo='refazer_captura' THEN 0 ELSE 1 END,
                r.created_at DESC
            """
        )

    def tem_revisao_pendente(self, imagem_id: int, tipo: str | None = None) -> bool:
        if tipo is None:
            row = self.db.fetchone(
                "SELECT id FROM revisao WHERE imagem_id=? AND resolvida=0 LIMIT 1",
                (imagem_id,),
            )
        else:
            row = self.db.fetchone(
                "SELECT id FROM revisao WHERE imagem_id=? AND tipo=? AND resolvida=0 LIMIT 1",
                (imagem_id, tipo),
            )
        return row is not None

    def resolver_revisao(self, revisao_id: int) -> None:
        revisao = self.db.fetchone("SELECT imagem_id FROM revisao WHERE id=?", (revisao_id,))
        self.db.update("UPDATE revisao SET resolvida=1 WHERE id=?", (revisao_id,))
        if revisao and revisao.get("imagem_id"):
            imagem_id = revisao["imagem_id"]
            if not self.tem_revisao_pendente(imagem_id):
                self.atualizar_imagem(imagem_id, precisa_revisao=0)

    def atualizar_revisao(self, revisao_id: int, detalhes: str) -> None:
        self.db.update(
            "UPDATE revisao SET detalhes=? WHERE id=?",
            (detalhes, revisao_id),
        )

    def salvar_sessao(self, **kwargs) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        kwargs["updated_at"] = now
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values())
        self.db.update(f"UPDATE sessao SET {sets} WHERE id=1", tuple(vals))

    def carregar_sessao(self) -> dict | None:
        row = self.db.fetchone("SELECT * FROM sessao WHERE id=1")
        if row and row.get("livro_id"):
            return row
        return None

    def get_total_imagens_livro(self, livro_id: int) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM imagem WHERE livro_id=?", (livro_id,))
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Registros civis e metadados pesquisáveis

    def sincronizar_registros_imagem(self, imagem_id: int) -> list[dict]:
        imagem = self.get_imagem(imagem_id)
        if not imagem:
            return []
        if imagem.get("tipo_documento", "registro") != "registro":
            return []
        if imagem.get("duplicidade_status") == "duplicata_confirmada":
            return []
        livro = self.get_livro(imagem["livro_id"]) or {}
        termo_i = imagem.get("termo_inicial")
        termo_f = imagem.get("termo_final")
        if termo_i is not None and termo_f is not None and termo_f >= termo_i:
            termos = list(range(int(termo_i), int(termo_f) + 1))
        else:
            quantidade = max(
                1,
                int(
                    imagem.get("registros_detectados")
                    or livro.get("registros_por_face")
                    or 1
                ),
            )
            termos = [None] * quantidade
        now = __import__("datetime").datetime.now().isoformat()
        for indice, termo in enumerate(termos):
            existente = self.db.fetchone(
                "SELECT id FROM registro WHERE imagem_id=? AND indice_na_imagem=?",
                (imagem_id, indice),
            )
            valores = (
                imagem["livro_id"], termo, imagem.get("folha_estimada"),
                imagem.get("face"), now,
            )
            if existente:
                self.db.update(
                    """
                    UPDATE registro
                    SET livro_id=?, termo=?, folha=?, face=?, updated_at=?
                    WHERE id=?
                    """,
                    valores + (existente["id"],),
                )
            else:
                self.db.insert(
                    """
                    INSERT INTO registro
                    (livro_id, imagem_id, indice_na_imagem, termo, folha, face,
                     status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'inferido', ?, ?)
                    """,
                    (
                        imagem["livro_id"], imagem_id, indice, termo,
                        imagem.get("folha_estimada"), imagem.get("face"), now, now,
                    ),
                )
        return self.listar_registros_imagem(imagem_id)

    def sincronizar_todos_registros(self) -> int:
        imagens = self.db.fetchall("SELECT id FROM imagem ORDER BY id")
        total = 0
        for imagem in imagens:
            total += len(self.sincronizar_registros_imagem(imagem["id"]))
        return total

    def listar_registros_imagem(self, imagem_id: int) -> list[dict]:
        return self.db.fetchall(
            "SELECT * FROM registro WHERE imagem_id=? ORDER BY indice_na_imagem",
            (imagem_id,),
        )

    def get_registro(self, registro_id: int) -> dict | None:
        return self.db.fetchone("SELECT * FROM registro WHERE id=?", (registro_id,))

    def get_execucao_ocr_ativa(
        self,
        *,
        imagem_id: int,
        registro_id: int | None,
        motor: str,
    ) -> dict | None:
        """Retorna a tentativa persistida, inclusive quando terminou em erro.

        Uma falha também é um resultado do processamento e não deve provocar
        repetição automática infinita. Nova tentativa só ocorre depois de uma
        invalidação explícita (por exemplo, quando a fotografia é substituída).
        """
        return self.db.fetchone(
            """
            SELECT * FROM ocr_execucao
            WHERE imagem_id=? AND motor=? AND ativo=1
              AND ((registro_id IS NULL AND ? IS NULL) OR registro_id=?)
            ORDER BY id DESC
            LIMIT 1
            """,
            (imagem_id, motor, registro_id, registro_id),
        )

    def listar_execucoes_ocr_ativas(
        self,
        *,
        imagem_id: int,
        registro_id: int | None,
    ) -> list[dict]:
        return self.db.fetchall(
            """
            SELECT * FROM ocr_execucao
            WHERE imagem_id=? AND ativo=1
              AND ((registro_id IS NULL AND ? IS NULL) OR registro_id=?)
            ORDER BY motor, id
            """,
            (imagem_id, registro_id, registro_id),
        )

    def invalidar_ocr_imagem(self, imagem_id: int) -> None:
        """Encerra os resultados da foto anterior preservando seu histórico."""
        self.db.update(
            "UPDATE ocr_execucao SET ativo=0 WHERE imagem_id=? AND ativo=1",
            (imagem_id,),
        )
        self.db.update(
            """
            UPDATE processamento_item
            SET status='pendente', tentativas=0, motor=NULL, resultado=NULL,
                confianca=0, tempo_ms=0, erro=NULL, iniciado_em=NULL,
                concluido_em=NULL, updated_at=datetime('now')
            WHERE imagem_id=? AND status!='confirmado'
            """,
            (imagem_id,),
        )
        self.db.update(
            """
            UPDATE ocr_deteccao
            SET ativo=0
            WHERE imagem_id=? AND ativo=1
              AND fonte != 'sequencia_livro'
            """,
            (imagem_id,),
        )
        self.db.update(
            """
            UPDATE revisao SET resolvida=1
            WHERE imagem_id=? AND resolvida=0
              AND tipo IN ('termo_incerto', 'folha_incerta', 'ocr_falha')
            """,
            (imagem_id,),
        )

    def criar_execucao_ocr(
        self,
        *,
        imagem_id: int,
        registro_id: int | None,
        motor: str,
        texto_bruto: str,
        tempo_ms: float = 0.0,
        sucesso: bool = True,
        erro: str = "",
        substituir_ativa: bool = True,
    ) -> int:
        # A versão anterior permanece no histórico, porém deixa de participar
        # das buscas correntes.
        if substituir_ativa:
            anteriores = self.db.fetchall(
                """
                SELECT id FROM ocr_execucao
                WHERE imagem_id=? AND motor=? AND ativo=1
                  AND ((registro_id IS NULL AND ? IS NULL) OR registro_id=?)
                """,
                (imagem_id, motor, registro_id, registro_id),
            )
            ids = [row["id"] for row in anteriores]
            for execucao_id in ids:
                self.db.update("UPDATE ocr_execucao SET ativo=0 WHERE id=?", (execucao_id,))
                self.db.update(
                    "UPDATE ocr_deteccao SET ativo=0 WHERE execucao_id=?",
                    (execucao_id,),
                )
        now = __import__("datetime").datetime.now().isoformat()
        return self.db.insert(
            """
            INSERT INTO ocr_execucao
            (imagem_id, registro_id, motor, texto_bruto, texto_normalizado,
             tempo_ms, sucesso, erro, ativo, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                imagem_id, registro_id, motor, texto_bruto,
                normalizar_busca(texto_bruto), float(tempo_ms),
                1 if sucesso else 0, erro, now,
            ),
        )

    def salvar_deteccoes_ocr(
        self,
        *,
        execucao_id: int,
        imagem_id: int,
        registro_id: int | None,
        deteccoes: list[dict],
    ) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        linhas = []
        for d in deteccoes:
            linhas.append((
                execucao_id, imagem_id, registro_id, d["tipo"],
                d.get("valor_original", ""), d.get("valor_tratado", ""),
                d.get("valor_normalizado", ""), float(d.get("confianca", 0)),
                d.get("motor", ""), d.get("fonte", "ocr"),
                d.get("status", "detectado"), d.get("bbox_json"),
                d.get("contexto", ""), now, now,
            ))
        self.db.executemany(
            """
            INSERT INTO ocr_deteccao
            (execucao_id, imagem_id, registro_id, tipo, valor_original,
             valor_tratado, valor_normalizado, confianca, motor, fonte,
             status, bbox_json, contexto, ativo, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            linhas,
        )

    def salvar_metadado_tratado(
        self,
        *,
        imagem_id: int,
        registro_id: int | None,
        tipo: str,
        valor: str,
        confianca: float,
        fonte: str,
        motor: str = "",
        status: str = "inferido",
        contexto: str = "",
    ) -> int:
        normalizado = normalizar_busca(valor)
        existente = self.db.fetchone(
            """
            SELECT id FROM ocr_deteccao
            WHERE imagem_id=? AND tipo=? AND fonte=? AND ativo=1
              AND ((registro_id IS NULL AND ? IS NULL) OR registro_id=?)
            LIMIT 1
            """,
            (imagem_id, tipo, fonte, registro_id, registro_id),
        )
        now = __import__("datetime").datetime.now().isoformat()
        if existente:
            self.db.update(
                """
                UPDATE ocr_deteccao
                SET valor_tratado=?, valor_normalizado=?, confianca=?, motor=?,
                    status=?, contexto=?, updated_at=?
                WHERE id=?
                """,
                (valor, normalizado, confianca, motor, status, contexto, now, existente["id"]),
            )
            deteccao_id = int(existente["id"])
        else:
            deteccao_id = self.db.insert(
                """
                INSERT INTO ocr_deteccao
                (execucao_id, imagem_id, registro_id, tipo, valor_original,
                 valor_tratado, valor_normalizado, confianca, motor, fonte,
                 status, contexto, ativo, created_at, updated_at)
                VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    imagem_id, registro_id, tipo, valor, valor, normalizado,
                    confianca, motor, fonte, status, contexto, now, now,
                ),
            )
        if (
            registro_id is not None
            and tipo == "nome_registrado"
            and status in {"confirmado", "corrigido"}
        ):
            self.marcar_registro_confirmado_processamento(int(registro_id))
        return deteccao_id

    def listar_metadados_registro(self, registro_id: int) -> list[dict]:
        registro = self.get_registro(registro_id)
        if not registro:
            return []
        return self.db.fetchall(
            """
            SELECT d.*,
                   CASE WHEN d.registro_id IS NULL THEN 'pagina' ELSE 'assento' END AS escopo
            FROM ocr_deteccao d
            WHERE d.ativo=1
              AND (d.registro_id=? OR (d.registro_id IS NULL AND d.imagem_id=?))
            ORDER BY
                CASE d.tipo WHEN 'nome_registrado' THEN 0 WHEN 'termo' THEN 1 ELSE 2 END,
                d.confianca DESC, d.id
            """,
            (registro_id, registro["imagem_id"]),
        )

    def listar_deteccoes_area(
        self,
        *,
        imagem_id: int,
        registro_id: int,
        tipo: str,
    ) -> list[dict]:
        return self.db.fetchall(
            """
            SELECT * FROM ocr_deteccao
            WHERE imagem_id=? AND registro_id=? AND tipo=?
              AND fonte='qwen_area' AND ativo=1
              AND bbox_json IS NOT NULL
            ORDER BY id DESC
            """,
            (imagem_id, registro_id, tipo),
        )

    def listar_execucoes_registro(self, registro_id: int) -> list[dict]:
        registro = self.get_registro(registro_id)
        if not registro:
            return []
        return self.db.fetchall(
            """
            SELECT e.*,
                   CASE WHEN e.registro_id IS NULL THEN 'pagina' ELSE 'assento' END AS escopo
            FROM ocr_execucao e
            WHERE e.ativo=1 AND e.sucesso=1
              AND (e.registro_id=? OR (e.registro_id IS NULL AND e.imagem_id=?))
            ORDER BY e.registro_id IS NULL, e.motor
            """,
            (registro_id, registro["imagem_id"]),
        )

    def confirmar_deteccao(self, deteccao_id: int) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        self.db.update(
            "UPDATE ocr_deteccao SET status='confirmado', updated_at=? WHERE id=?",
            (now, deteccao_id),
        )
        row = self.db.fetchone(
            "SELECT registro_id, tipo FROM ocr_deteccao WHERE id=?", (deteccao_id,)
        )
        if row and row.get("registro_id") and row.get("tipo") == "nome_registrado":
            self.marcar_registro_confirmado_processamento(int(row["registro_id"]))

    def corrigir_deteccao(self, deteccao_id: int, novo_valor: str) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        self.db.update(
            """
            UPDATE ocr_deteccao
            SET valor_tratado=?, valor_normalizado=?, status='corrigido',
                updated_at=?
            WHERE id=?
            """,
            (novo_valor, normalizar_busca(novo_valor), now, deteccao_id),
        )
        row = self.db.fetchone(
            "SELECT registro_id, tipo FROM ocr_deteccao WHERE id=?", (deteccao_id,)
        )
        if row and row.get("registro_id") and row.get("tipo") == "nome_registrado":
            self.marcar_registro_confirmado_processamento(int(row["registro_id"]))

    # ------------------------------------------------------------------
    # Fila persistente de nomes

    def criar_ou_sincronizar_lote_nomes(self, livro_id: int) -> dict:
        chave = f"nomes-v2-livro-{int(livro_id)}"
        lote = self.db.fetchone(
            "SELECT * FROM processamento_lote WHERE chave=?", (chave,)
        )
        now = __import__("datetime").datetime.now().isoformat()
        if lote is None:
            lote_id = self.db.insert(
                """
                INSERT INTO processamento_lote
                (chave, tipo, livro_id, status, created_at, updated_at)
                VALUES (?, 'nomes_registros', ?, 'pendente', ?, ?)
                """,
                (chave, int(livro_id), now, now),
            )
        else:
            lote_id = int(lote["id"])

        registros = self.db.fetchall(
            """
            SELECT r.id AS registro_id, r.imagem_id,
                   COALESCE(i.sha256_normalizado, i.sha256_armazenamento, i.sha256) AS sha256,
                   i.orientacao_confianca,
                   CASE WHEN EXISTS (
                       SELECT 1 FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status IN ('confirmado', 'corrigido')
                   ) THEN 1 ELSE 0 END AS nome_confirmado
            FROM registro r
            JOIN imagem i ON i.id=r.imagem_id
            WHERE r.livro_id=?
              AND COALESCE(i.duplicidade_status, '')!='duplicata_confirmada'
              AND COALESCE(i.tipo_documento, 'registro')='registro'
            ORDER BY r.termo, r.id
            """,
            (int(livro_id),),
        )
        linhas = [
            (
                lote_id, int(livro_id), int(item["imagem_id"]),
                int(item["registro_id"]), item.get("sha256"),
                (
                    "confirmado" if item.get("nome_confirmado") else
                    "pausado" if (
                        item.get("orientacao_confianca") is not None
                        and float(item.get("orientacao_confianca") or 0) < 0.85
                    ) else "pendente"
                ),
                now, now,
            )
            for item in registros
        ]
        self.db.executemany(
            """
            INSERT OR IGNORE INTO processamento_item
            (lote_id, livro_id, imagem_id, registro_id, imagem_sha256,
             etapa, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'ocr_nome_rapido', ?, ?, ?)
            """,
            linhas,
        )
        # Uma confirmação humana feita depois da criação do lote sempre vence.
        self.db.update(
            """
            UPDATE processamento_item
            SET status='confirmado', erro=NULL, concluido_em=COALESCE(concluido_em, ?),
                updated_at=?
            WHERE lote_id=? AND EXISTS (
                SELECT 1 FROM ocr_deteccao d
                WHERE d.registro_id=processamento_item.registro_id AND d.ativo=1
                  AND d.tipo='nome_registrado'
                  AND d.status IN ('confirmado', 'corrigido')
            )
            """,
            (now, now, lote_id),
        )
        # Ao confirmar/corrigir a orientação no Revisor, a sincronização torna
        # o item elegível sem recriá-lo nem perder histórico.
        self.db.update(
            """
            UPDATE processamento_item
            SET status='pendente', updated_at=?
            WHERE lote_id=? AND etapa='ocr_nome_rapido' AND status='pausado'
              AND EXISTS (
                SELECT 1 FROM imagem i WHERE i.id=processamento_item.imagem_id
                  AND (i.orientacao_confianca IS NULL OR i.orientacao_confianca>=0.85)
              )
            """,
            (now, lote_id),
        )
        self.db.update(
            "UPDATE processamento_lote SET total_registros=?, updated_at=? WHERE id=?",
            (len(registros), now, lote_id),
        )
        return self.db.fetchone(
            "SELECT * FROM processamento_lote WHERE id=?", (lote_id,)
        ) or {"id": lote_id, "livro_id": livro_id}

    def preparar_retomada_lote(self, lote_id: int) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        self.db.update(
            """
            UPDATE processamento_item
            SET status='pendente', iniciado_em=NULL, updated_at=?
            WHERE lote_id=? AND status='processando'
            """,
            (now, int(lote_id)),
        )

    def marcar_lote_status(self, lote_id: int, status: str) -> None:
        now = __import__("datetime").datetime.now().isoformat()
        inicio = now if status == "processando" else None
        conclusao = now if status == "concluido" else None
        self.db.update(
            """
            UPDATE processamento_lote
            SET status=?, iniciado_em=COALESCE(iniciado_em, ?),
                concluido_em=CASE WHEN ? IS NULL THEN concluido_em ELSE ? END,
                updated_at=? WHERE id=?
            """,
            (status, inicio, conclusao, conclusao, now, int(lote_id)),
        )

    def listar_itens_processamento(
        self,
        lote_id: int,
        *,
        etapa: str | None = None,
        statuses: tuple[str, ...] | None = None,
        limite: int = 5000,
    ) -> list[dict]:
        condicoes = ["p.lote_id=?"]
        params: list[Any] = [int(lote_id)]
        if etapa:
            condicoes.append("p.etapa=?")
            params.append(etapa)
        if statuses:
            condicoes.append(f"p.status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        params.append(max(1, min(int(limite), 10000)))
        return self.db.fetchall(
            f"""
            SELECT p.*, r.termo, r.folha, r.face, r.indice_na_imagem,
                   i.caminho_original,
                   i.caminho_normalizado,
                   COALESCE(i.caminho_normalizado, i.caminho_armazenamento) AS caminho_armazenamento,
                   i.rotacao_visualizacao,
                   i.sha256 AS sha256_atual,
                   i.sha256_armazenamento,
                   i.termo_inicial, i.termo_final, l.codigo AS livro_codigo,
                   (SELECT COUNT(*) FROM registro rr WHERE rr.imagem_id=i.id) AS total_na_imagem
            FROM processamento_item p
            JOIN registro r ON r.id=p.registro_id
            JOIN imagem i ON i.id=p.imagem_id
            JOIN livro l ON l.id=p.livro_id
            WHERE {' AND '.join(condicoes)}
            ORDER BY r.termo, p.etapa, p.id
            LIMIT ?
            """,
            tuple(params),
        )

    def atualizar_item_processamento(self, item_id: int, **kwargs) -> None:
        kwargs["updated_at"] = __import__("datetime").datetime.now().isoformat()
        sets = ", ".join(f"{campo}=?" for campo in kwargs)
        self.db.update(
            f"UPDATE processamento_item SET {sets} WHERE id=?",
            tuple(kwargs.values()) + (int(item_id),),
        )

    def garantir_item_qwen(self, item: dict, *, status: str = "pendente") -> int:
        now = __import__("datetime").datetime.now().isoformat()
        existente = self.db.fetchone(
            """
            SELECT id FROM processamento_item
            WHERE lote_id=? AND registro_id=? AND etapa='qwen_nome'
            """,
            (int(item["lote_id"]), int(item["registro_id"])),
        )
        if existente:
            return int(existente["id"])
        return self.db.insert(
            """
            INSERT INTO processamento_item
            (lote_id, livro_id, imagem_id, registro_id, imagem_sha256,
             etapa, status, bbox_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'qwen_nome', ?, ?, ?, ?)
            """,
            (
                int(item["lote_id"]), int(item["livro_id"]),
                int(item["imagem_id"]), int(item["registro_id"]),
                item.get("imagem_sha256"), status, item.get("bbox_json"), now, now,
            ),
        )

    def reprocessar_falhas_lote(self, lote_id: int) -> int:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM processamento_item WHERE lote_id=? AND status='falhou'",
            (int(lote_id),),
        ) or {"n": 0}
        self.db.update(
            """
            UPDATE processamento_item
            SET status='pendente', tentativas=0, erro=NULL, iniciado_em=NULL,
                concluido_em=NULL, updated_at=datetime('now')
            WHERE lote_id=? AND status='falhou'
            """,
            (int(lote_id),),
        )
        return int(row["n"])

    def reabrir_qwen_para_termos(
        self, livro_id: int, termos: list[int] | tuple[int, ...], *, motivo: str
    ) -> int:
        """Reabre sugestões Qwen sem apagar o histórico nem confirmações humanas."""
        valores = sorted({int(termo) for termo in termos})
        if not valores:
            return 0
        placeholders = ",".join("?" for _ in valores)
        registros = self.db.fetchall(
            f"SELECT id FROM registro WHERE livro_id=? AND termo IN ({placeholders})",
            (int(livro_id), *valores),
        )
        ids = [int(row["id"]) for row in registros]
        if not ids:
            return 0
        id_placeholders = ",".join("?" for _ in ids)
        now = __import__("datetime").datetime.now().isoformat()
        self.db.update(
            f"""
            UPDATE ocr_deteccao
            SET ativo=0, status='descartado', updated_at=?,
                contexto=COALESCE(contexto,'') || ?
            WHERE registro_id IN ({id_placeholders}) AND ativo=1
              AND fonte='qwen_nome_correcao'
              AND status NOT IN ('confirmado','corrigido')
            """,
            (now, f" | Reprocessado: {motivo[:300]}", *ids),
        )
        self.db.update(
            f"""
            UPDATE ocr_execucao SET ativo=0
            WHERE registro_id IN ({id_placeholders})
              AND motor LIKE 'qwen-nome-faixa%'
            """,
            tuple(ids),
        )
        self.db.update(
            f"""
            UPDATE processamento_item
            SET status='pendente', tentativas=0, resultado=NULL, confianca=0,
                tempo_ms=0, erro=NULL, iniciado_em=NULL, concluido_em=NULL,
                updated_at=?
            WHERE livro_id=? AND etapa='qwen_nome'
              AND registro_id IN ({id_placeholders})
              AND status!='confirmado'
            """,
            (now, int(livro_id), *ids),
        )
        return len(ids)

    def marcar_registro_confirmado_processamento(self, registro_id: int) -> None:
        self.db.update(
            """
            UPDATE processamento_item
            SET status='confirmado', erro=NULL,
                concluido_em=COALESCE(concluido_em, datetime('now')),
                updated_at=datetime('now')
            WHERE registro_id=?
            """,
            (int(registro_id),),
        )

    def resumo_processamento(self, lote_id: int) -> dict:
        lote = self.db.fetchone(
            "SELECT * FROM processamento_lote WHERE id=?", (int(lote_id),)
        ) or {}
        linhas = self.db.fetchall(
            """
            SELECT etapa, status, COUNT(*) AS n, AVG(NULLIF(tempo_ms, 0)) AS media_ms
            FROM processamento_item WHERE lote_id=? GROUP BY etapa, status
            """,
            (int(lote_id),),
        )
        contagens = {
            f"{row['etapa']}:{row['status']}": int(row["n"])
            for row in linhas
        }
        medias = self.db.fetchall(
            """
            SELECT etapa, AVG(tempo_ms) AS media_ms
            FROM processamento_item
            WHERE lote_id=? AND tempo_ms>0 GROUP BY etapa
            """,
            (int(lote_id),),
        )
        lote["medias_ms"] = {
            str(row["etapa"]): float(row.get("media_ms") or 0)
            for row in medias
        }
        lote["contagens"] = contagens
        lote["itens"] = sum(int(row["n"]) for row in linhas)
        lote["confirmados"] = int((self.db.fetchone(
            """
            SELECT COUNT(DISTINCT d.registro_id) AS n
            FROM ocr_deteccao d JOIN registro r ON r.id=d.registro_id
            WHERE r.livro_id=? AND d.ativo=1 AND d.tipo='nome_registrado'
              AND d.status IN ('confirmado', 'corrigido')
            """,
            (lote.get("livro_id"),),
        ) or {"n": 0})["n"])
        return lote

    def listar_associacoes_qwen_invalidas(
        self, livro_id: int | None = None
    ) -> list[dict]:
        """Localiza leituras cuja caixa não pertence ao assento associado."""
        params: tuple[Any, ...] = ()
        filtro = ""
        if livro_id is not None:
            filtro = "AND r.livro_id=?"
            params = (int(livro_id),)
        deteccoes = self.db.fetchall(
            f"""
            SELECT d.id, d.execucao_id, d.registro_id, d.imagem_id, d.bbox_json,
                   d.valor_tratado, r.termo, r.indice_na_imagem,
                   (SELECT COUNT(*) FROM registro rr WHERE rr.imagem_id=r.imagem_id) AS total
            FROM ocr_deteccao d
            JOIN registro r ON r.id=d.registro_id
            WHERE d.ativo=1 AND d.fonte='qwen_registro'
              AND d.bbox_json IS NOT NULL {filtro}
            ORDER BY d.id
            """,
            params,
        )
        invalidas: list[dict] = []
        for item in deteccoes:
            try:
                bbox = json.loads(item.get("bbox_json") or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                bbox = []
            if not bbox_corresponde_registro(
                bbox,
                int(item.get("indice_na_imagem") or 0),
                int(item.get("total") or 1),
            ):
                invalidas.append(item)
        return invalidas

    def auditar_associacoes_qwen(self, livro_id: int | None = None) -> list[dict]:
        """Desativa leituras automáticas associadas ao assento errado."""
        invalidas = self.listar_associacoes_qwen_invalidas(livro_id=livro_id)
        execucoes = {
            int(item["execucao_id"])
            for item in invalidas
            if item.get("execucao_id")
        }
        for execucao_id in execucoes:
            self.db.update(
                "UPDATE ocr_execucao SET ativo=0, erro=COALESCE(NULLIF(erro,''), 'associação de registro inválida') WHERE id=?",
                (execucao_id,),
            )
            self.db.update(
                """
                UPDATE ocr_deteccao
                SET ativo=0, status='descartado',
                    contexto=COALESCE(contexto, '') || ' | Descartado: caixa incompatível com o assento.',
                    updated_at=datetime('now')
                WHERE execucao_id=? AND ativo=1
                """,
                (execucao_id,),
            )
        for item in invalidas:
            if not self.tem_revisao_pendente(int(item["imagem_id"]), "qwen_associacao_invalida"):
                self.criar_revisao(
                    imagem_id=int(item["imagem_id"]),
                    tipo="qwen_associacao_invalida",
                    detalhes=(
                        f"Termo {item.get('termo')}: leitura antiga preservada, mas "
                        "desativada porque o recorte pertence a outro assento."
                    ),
                )
        return invalidas

    def rebaixar_sugestoes_rapidas_antigas(self, livro_id: int | None = None) -> int:
        """Retira da pesquisa sugestões criadas pelo critério antigo.

        Antes da calibração, qualquer nome extraído pelo Tesseract recebia
        0,72 e podia aparecer como sugestão. Essas detecções permanecem no
        histórico, mas não podem continuar parecendo nomes plausíveis. O
        marcador `concordancia_motores` protege as execuções novas.
        """
        filtro = ""
        params: tuple = ()
        if livro_id is not None:
            filtro = "AND r.livro_id=?"
            params = (int(livro_id),)
        linhas = self.db.fetchall(
            f"""
            SELECT d.id, d.imagem_id, d.registro_id
            FROM ocr_deteccao d
            JOIN registro r ON r.id=d.registro_id
            WHERE d.ativo=1 AND d.status='sugestao'
              AND d.fonte='ocr_nome_rapido'
              AND COALESCE(d.contexto, '') NOT LIKE '%concordancia_motores=%'
              {filtro}
            """,
            params,
        )
        for linha in linhas:
            self.db.update(
                """
                UPDATE ocr_deteccao
                SET ativo=0, status='descartado',
                    contexto=COALESCE(contexto, '') ||
                      ' | Rebaixado: OCR rápido antigo sem concordância entre motores.',
                    updated_at=datetime('now')
                WHERE id=?
                """,
                (int(linha["id"]),),
            )
            self.db.update(
                """
                UPDATE processamento_item
                SET status='revisar', updated_at=datetime('now')
                WHERE registro_id=? AND etapa='ocr_nome_rapido'
                  AND status IN ('sugestao', 'pendente')
                """,
                (int(linha["registro_id"]),),
            )
        return len(linhas)

    def listar_acervo_livros(self) -> list[dict]:
        return self.db.fetchall(
            """
            SELECT l.*, a.nome AS acervo_nome, o.nome AS oficio_nome,
                   t.nome AS tipo_nome,
                   (SELECT COUNT(*) FROM imagem i WHERE i.livro_id=l.id) AS total_imagens,
                   (SELECT COUNT(*) FROM registro r WHERE r.livro_id=l.id) AS total_registros,
                   CASE WHEN l.termo_inicial IS NOT NULL AND l.termo_final IS NOT NULL
                        THEN l.termo_final-l.termo_inicial+1
                        ELSE COALESCE(l.total_folhas, 0) * COALESCE(l.registros_por_face, 1)
                             * CASE WHEN COALESCE(l.frente_verso, 0)=1 THEN 2 ELSE 1 END
                   END AS total_esperado,
                   (SELECT COUNT(*) FROM ocorrencia oc
                    WHERE oc.livro_id=l.id AND oc.tipo='face_ausente') AS faces_faltantes,
                    (SELECT COUNT(DISTINCT p.registro_id) FROM processamento_item p
                     WHERE p.livro_id=l.id AND (
                       (p.etapa='qwen_nome' AND p.status IN ('revisar','sem_resultado','falhou','confirmado'))
                       OR (p.etapa='ocr_nome_rapido' AND p.status IN ('sugestao','revisar','sem_resultado','falhou','confirmado'))
                     )) AS nomes_processados,
                    (SELECT COUNT(DISTINCT p.registro_id) FROM processamento_item p
                     WHERE p.livro_id=l.id AND (
                       (p.etapa='qwen_nome' AND p.status IN ('pendente','processando','pausado'))
                       OR (p.etapa='ocr_nome_rapido' AND p.status IN ('pendente','pausado','processando'))
                     )) AS nomes_pendentes,
                    (SELECT COUNT(DISTINCT d.registro_id) FROM ocr_deteccao d
                     JOIN registro rr ON rr.id=d.registro_id
                     WHERE rr.livro_id=l.id AND d.ativo=1 AND d.tipo='nome_registrado'
                       AND d.status NOT IN ('confirmado','corrigido','descartado')) AS nomes_sugestoes,
                    (SELECT COUNT(DISTINCT p.registro_id) FROM processamento_item p
                     WHERE p.livro_id=l.id AND p.status IN ('revisar','falhou','sem_resultado')) AS nomes_revisao,
                    (SELECT COUNT(DISTINCT d.registro_id) FROM ocr_deteccao d
                     JOIN registro rr ON rr.id=d.registro_id
                     WHERE rr.livro_id=l.id AND d.ativo=1 AND d.tipo='nome_registrado'
                       AND d.status IN ('confirmado','corrigido')) AS nomes_confirmados
            FROM livro l
            LEFT JOIN acervo a ON a.id=l.acervo_id
            JOIN oficio o ON o.id=l.oficio_id
            JOIN tipo_registro t ON t.id=l.tipo_id
            ORDER BY a.nome, o.nome, t.nome, l.codigo
            """
        )

    def buscar_registros(
        self,
        *,
        texto: str = "",
        termo: int | None = None,
        acervo_id: int | None = None,
        oficio_id: int | None = None,
        tipo_id: int | None = None,
        livro_id: int | None = None,
        limite: int = 300,
    ) -> list[dict]:
        condicoes = ["i.duplicidade_status != 'duplicata_confirmada'"]
        params: list[Any] = []
        if termo is not None:
            condicoes.append("r.termo=?")
            params.append(int(termo))
        if acervo_id is not None:
            condicoes.append("l.acervo_id=?")
            params.append(acervo_id)
        if oficio_id is not None:
            condicoes.append("l.oficio_id=?")
            params.append(oficio_id)
        if tipo_id is not None:
            condicoes.append("l.tipo_id=?")
            params.append(tipo_id)
        if livro_id is not None:
            condicoes.append("l.id=?")
            params.append(livro_id)
        consulta = normalizar_busca(texto)
        if consulta:
            like = f"%{consulta}%"
            condicoes.append(
                """
                (
                    UPPER(REPLACE(REPLACE(l.codigo, '-', ' '), '/', ' ')) LIKE ? OR
                    EXISTS (
                        SELECT 1 FROM ocr_deteccao d
                        WHERE d.ativo=1 AND d.valor_normalizado LIKE ?
                          AND (d.registro_id=r.id OR (d.registro_id IS NULL AND d.imagem_id=i.id))
                    ) OR
                    EXISTS (
                        SELECT 1 FROM ocr_execucao e
                        WHERE e.ativo=1 AND e.imagem_id=i.id
                          AND (e.registro_id IS NULL OR e.registro_id=r.id)
                          AND e.texto_normalizado LIKE ?
                    )
                )
                """
            )
            params.extend([like, like, like])
        sql = f"""
            SELECT r.id AS registro_id, r.termo, r.folha, r.face,
                   r.indice_na_imagem, r.status AS registro_status,
                   i.id AS imagem_id, i.caminho_original, i.caminho_thumb,
                   i.sha256,
                   i.qualidade_status, i.termo_inicial, i.termo_final,
                   i.rotacao_visualizacao, i.tipo_documento,
                   i.ocr_termo, i.termo_status, i.confianca_termo,
                   l.id AS livro_id, l.codigo AS livro_codigo,
                   l.nome_capa AS livro_nome, a.nome AS acervo_nome,
                   o.nome AS oficio_nome, t.nome AS tipo_nome,
                   COALESCE((
                       SELECT d.valor_tratado
                       FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status IN ('confirmado', 'corrigido')
                       ORDER BY CASE d.status WHEN 'corrigido' THEN 0 ELSE 1 END,
                                d.id DESC LIMIT 1
                   ), '') AS nome_confirmado,
                   COALESCE((
                       SELECT d.valor_tratado
                       FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status NOT IN ('confirmado', 'corrigido', 'superado', 'descartado')
                       ORDER BY CASE d.fonte
                                  WHEN 'qwen_nome_correcao' THEN 0
                                  WHEN 'qwen_registro' THEN 1
                                  WHEN 'ocr_nome_rapido' THEN 2
                                  ELSE 3 END,
                                d.confianca DESC, d.id DESC LIMIT 1
                   ), '') AS nome_sugerido,
                   COALESCE((
                       SELECT d.status FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status NOT IN ('superado', 'descartado')
                       ORDER BY CASE WHEN d.status IN ('confirmado', 'corrigido') THEN 0 ELSE 1 END,
                                CASE d.fonte WHEN 'qwen_nome_correcao' THEN 0 WHEN 'qwen_registro' THEN 1 ELSE 2 END,
                                d.confianca DESC, d.id DESC LIMIT 1
                   ), '') AS nome_status,
                   COALESCE((
                       SELECT d.confianca FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status NOT IN ('superado', 'descartado')
                       ORDER BY CASE WHEN d.status IN ('confirmado', 'corrigido') THEN 0 ELSE 1 END,
                                CASE d.fonte WHEN 'qwen_nome_correcao' THEN 0 WHEN 'qwen_registro' THEN 1 ELSE 2 END,
                                d.confianca DESC, d.id DESC LIMIT 1
                   ), 0) AS nome_confianca,
                   COALESCE((
                       SELECT d.fonte FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status NOT IN ('superado', 'descartado')
                       ORDER BY CASE WHEN d.status IN ('confirmado', 'corrigido') THEN 0 ELSE 1 END,
                                CASE d.fonte WHEN 'qwen_nome_correcao' THEN 0 WHEN 'qwen_registro' THEN 1 ELSE 2 END,
                                d.confianca DESC, d.id DESC LIMIT 1
                   ), '') AS nome_fonte,
                   COALESCE((
                       SELECT d.valor_tratado FROM ocr_deteccao d
                       WHERE d.registro_id=r.id AND d.ativo=1
                         AND d.tipo='nome_registrado'
                         AND d.status NOT IN ('superado', 'descartado')
                       ORDER BY CASE WHEN d.status IN ('confirmado', 'corrigido') THEN 0 ELSE 1 END,
                                CASE d.fonte WHEN 'qwen_nome_correcao' THEN 0 WHEN 'qwen_registro' THEN 1 ELSE 2 END,
                                d.confianca DESC, d.id DESC LIMIT 1
                   ), '') AS nomes
            FROM registro r
            JOIN imagem i ON i.id=r.imagem_id
            JOIN livro l ON l.id=r.livro_id
            LEFT JOIN acervo a ON a.id=l.acervo_id
            JOIN oficio o ON o.id=l.oficio_id
            JOIN tipo_registro t ON t.id=l.tipo_id
            WHERE {' AND '.join(condicoes)}
            ORDER BY a.nome, o.nome, l.codigo, r.termo, r.id
            LIMIT ?
        """
        params.append(max(1, min(int(limite), 2000)))
        rows = self.db.fetchall(sql, tuple(params))
        if not consulta:
            return rows

        # A busca SQL continua sendo a primeira fonte (livro, OCR e texto
        # exato). Quando uma grafia manuscrita diverge, acrescentamos uma
        # segunda camada pequena de Levenshtein sobre os nomes ativos. A lista
        # sem filtro tem no máximo 2.000 registros e o resultado é marcado
        # explicitamente como aproximação para nunca parecer confirmação.
        candidatos = self.buscar_registros(
            texto="",
            termo=termo,
            acervo_id=acervo_id,
            oficio_id=oficio_id,
            tipo_id=tipo_id,
            livro_id=livro_id,
            limite=2000,
        )
        ids_exatos = {int(row["registro_id"]) for row in rows}
        fuzzy: list[dict] = []
        for candidato in candidatos:
            if int(candidato["registro_id"]) in ids_exatos:
                continue
            nome = (
                candidato.get("nome_confirmado")
                or candidato.get("nome_sugerido")
                or candidato.get("nomes")
                or ""
            )
            if len(_nome_normalizado_fuzzy(consulta)) < 4 or not nome:
                continue
            score = _similaridade_levenshtein(consulta, str(nome))
            # Nomes manuscritos podem perder uma palavra curta ou duas letras,
            # mas um resultado abaixo disso já produz falsos positivos demais.
            if score < 0.70:
                continue
            item = dict(candidato)
            item["nome_busca_similaridade"] = round(score, 4)
            item["busca_fuzzy"] = True
            fuzzy.append(item)
        fuzzy.sort(key=lambda row: float(row["nome_busca_similaridade"]), reverse=True)
        disponivel = max(0, int(limite) - len(rows))
        return rows + fuzzy[:disponivel]

    def listar_imagens_para_indexacao(
        self,
        livro_id: int | None = None,
        somente_pendentes: bool = True,
        motores_pendentes: list[str] | None = None,
    ) -> list[dict]:
        condicoes = ["COALESCE(i.duplicidade_status, '') != 'duplicata_confirmada'"]
        params: list[Any] = []
        if livro_id is not None:
            condicoes.append("i.livro_id=?")
            params.append(livro_id)
        motores = [m.strip() for m in (motores_pendentes or []) if m.strip()]
        if motores:
            faltantes = []
            for motor in motores:
                faltantes.append(
                    """
                    NOT EXISTS (
                        SELECT 1 FROM ocr_execucao e
                        WHERE e.imagem_id=i.id AND e.ativo=1 AND e.motor=?
                          AND (e.registro_id=r.id OR e.registro_id IS NULL)
                    )
                    """
                )
                params.append(motor)
            condicoes.append(
                f"""
                EXISTS (
                    SELECT 1 FROM registro r
                    WHERE r.imagem_id=i.id AND ({' OR '.join(faltantes)})
                )
                """
            )
        elif somente_pendentes:
            condicoes.append(
                """
                EXISTS (
                    SELECT 1 FROM registro r
                    WHERE r.imagem_id=i.id
                      AND NOT EXISTS (
                          SELECT 1 FROM ocr_execucao e
                          WHERE e.imagem_id=i.id AND e.ativo=1
                            AND (e.registro_id=r.id OR e.registro_id IS NULL)
                      )
                )
                """
            )
        return self.db.fetchall(
            f"""
            SELECT i.*, l.codigo AS livro_codigo
            FROM imagem i
            JOIN livro l ON l.id=i.livro_id
            WHERE {' AND '.join(condicoes)}
            ORDER BY l.codigo, i.ordem_captura
            """,
            tuple(params),
        )

    def estatisticas_consulta(self) -> dict:
        return {
            "acervos": (self.db.fetchone("SELECT COUNT(*) cnt FROM acervo WHERE ativo=1") or {"cnt": 0})["cnt"],
            "livros": (self.db.fetchone("SELECT COUNT(*) cnt FROM livro") or {"cnt": 0})["cnt"],
            "imagens": (self.db.fetchone("SELECT COUNT(*) cnt FROM imagem") or {"cnt": 0})["cnt"],
            "registros": (self.db.fetchone("SELECT COUNT(*) cnt FROM registro") or {"cnt": 0})["cnt"],
            "nomes": (self.db.fetchone("SELECT COUNT(*) cnt FROM ocr_deteccao WHERE tipo='nome_registrado' AND ativo=1") or {"cnt": 0})["cnt"],
        }
