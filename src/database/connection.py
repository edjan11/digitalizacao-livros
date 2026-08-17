from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from ..config.settings import APP_VERSION, data_dir


# Incrementar somente quando SCHEMA/_migrate/SEEDS exigirem uma nova passagem.
# Abrir uma conexao de worker nao deve recriar dezenas de tabelas e indices.
SCHEMA_REVISION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS acervo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT UNIQUE NOT NULL,
    descricao TEXT,
    caminho_root TEXT,
    ativo INTEGER DEFAULT 1,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS oficio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT UNIQUE NOT NULL,
    ativo INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tipo_registro (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    subtipo_de INTEGER REFERENCES tipo_registro(id),
    ativo INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS livro (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    acervo_id INTEGER DEFAULT 1 REFERENCES acervo(id),
    oficio_id INTEGER NOT NULL REFERENCES oficio(id),
    tipo_id INTEGER NOT NULL REFERENCES tipo_registro(id),
    subtipo_id INTEGER REFERENCES tipo_registro(id),
    codigo TEXT,
    nome_capa TEXT,
    total_folhas INTEGER,
    primeira_folha INTEGER,
    ultima_folha INTEGER,
    frente_verso INTEGER DEFAULT 1,
    registros_por_face INTEGER DEFAULT 1,
    termo_inicial INTEGER,
    termo_final INTEGER,
    registros_detectados INTEGER,
    layout_id TEXT,
    layout_confidence REAL,
    layout_method TEXT,
    layout_reason TEXT,
    observacoes TEXT,
    status TEXT DEFAULT 'em_andamento',
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS imagem (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    livro_id INTEGER NOT NULL REFERENCES livro(id),
    ordem_captura INTEGER NOT NULL,
    hash_perceptual TEXT,
    dhash TEXT,
    sha256 TEXT,
    caminho_original TEXT,
    caminho_armazenamento TEXT,
    sha256_armazenamento TEXT,
    caminho_normalizado TEXT,
    sha256_normalizado TEXT,
    orientacao_confianca REAL,
    orientacao_metodo TEXT,
    orientacao_motivo TEXT,
    normalizacao_json TEXT,
    caminho_thumb TEXT,
    tipo_documento TEXT DEFAULT 'registro',
    rotacao_visualizacao INTEGER DEFAULT 0,
    origem_posicao INTEGER,
    folha_estimada INTEGER,
    face TEXT CHECK(face IN ('frente','verso','indeterminado')),
    folha_status TEXT DEFAULT 'nao_identificado',
    termo_inicial INTEGER,
    termo_final INTEGER,
    qualidade_foco REAL,
    qualidade_exposicao REAL,
    qualidade_enquadramento TEXT,
    qualidade_orientacao INTEGER DEFAULT 0,
    qualidade_status TEXT DEFAULT 'pendente',
    qualidade_oclusao REAL DEFAULT 0,
    qualidade_motivos TEXT,
    duplicidade_status TEXT DEFAULT 'pendente',
    duplicidade_confianca REAL,
    duplicidade_ref INTEGER REFERENCES imagem(id),
    ocr_termo TEXT,
    ocr_folha TEXT,
    ocr_confianca REAL,
    htr_termo TEXT,
    htr_folha TEXT,
    htr_confianca REAL,
    termo_final_decidido INTEGER,
    folha_final_decidida INTEGER,
    termo_status TEXT DEFAULT 'pendente',
    motor_utilizado TEXT,
    confianca_termo REAL,
    confianca_folha REAL,
    status TEXT DEFAULT 'pendente',
    precisa_revisao INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS registro (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    livro_id INTEGER NOT NULL REFERENCES livro(id),
    imagem_id INTEGER NOT NULL REFERENCES imagem(id),
    indice_na_imagem INTEGER NOT NULL DEFAULT 0,
    termo INTEGER,
    folha INTEGER,
    face TEXT,
    status TEXT DEFAULT 'inferido',
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(imagem_id, indice_na_imagem)
);

CREATE TABLE IF NOT EXISTS ocr_execucao (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    imagem_id INTEGER NOT NULL REFERENCES imagem(id),
    registro_id INTEGER REFERENCES registro(id),
    motor TEXT NOT NULL,
    texto_bruto TEXT,
    texto_normalizado TEXT,
    tempo_ms REAL DEFAULT 0,
    sucesso INTEGER DEFAULT 1,
    erro TEXT,
    ativo INTEGER DEFAULT 1,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS ocr_deteccao (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execucao_id INTEGER REFERENCES ocr_execucao(id),
    imagem_id INTEGER NOT NULL REFERENCES imagem(id),
    registro_id INTEGER REFERENCES registro(id),
    tipo TEXT NOT NULL,
    valor_original TEXT,
    valor_tratado TEXT,
    valor_normalizado TEXT,
    confianca REAL DEFAULT 0,
    motor TEXT,
    fonte TEXT DEFAULT 'ocr',
    status TEXT DEFAULT 'detectado',
    bbox_json TEXT,
    contexto TEXT,
    ativo INTEGER DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS ocorrencia (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    livro_id INTEGER NOT NULL REFERENCES livro(id),
    tipo TEXT NOT NULL,
    folha_afetada INTEGER,
    termo_afetado INTEGER,
    descricao TEXT,
    confirmada INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS revisao (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    imagem_id INTEGER REFERENCES imagem(id),
    tipo TEXT NOT NULL,
    detalhes TEXT,
    resolvida INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS sessao (
    id INTEGER PRIMARY KEY DEFAULT 1,
    oficio_id INTEGER,
    tipo_id INTEGER,
    subtipo_id INTEGER,
    livro_id INTEGER,
    ultima_folha INTEGER,
    ultima_face TEXT,
    ultimo_termo INTEGER,
    pasta_monitorada TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS processamento_lote (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chave TEXT UNIQUE NOT NULL,
    tipo TEXT NOT NULL,
    livro_id INTEGER NOT NULL REFERENCES livro(id),
    status TEXT NOT NULL DEFAULT 'pendente',
    total_registros INTEGER DEFAULT 0,
    iniciado_em TEXT,
    concluido_em TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS processamento_item (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lote_id INTEGER NOT NULL REFERENCES processamento_lote(id),
    livro_id INTEGER NOT NULL REFERENCES livro(id),
    imagem_id INTEGER NOT NULL REFERENCES imagem(id),
    registro_id INTEGER NOT NULL REFERENCES registro(id),
    imagem_sha256 TEXT,
    etapa TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pendente',
    tentativas INTEGER DEFAULT 0,
    motor TEXT,
    resultado TEXT,
    confianca REAL DEFAULT 0,
    tempo_ms REAL DEFAULT 0,
    bbox_json TEXT,
    erro TEXT,
    iniciado_em TEXT,
    concluido_em TEXT,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(lote_id, registro_id, etapa)
);

CREATE TABLE IF NOT EXISTS importacao_lote (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chave TEXT UNIQUE NOT NULL,
    livro_id INTEGER REFERENCES livro(id),
    codigo_livro TEXT NOT NULL,
    raiz TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'auditoria',
    total_arquivos INTEGER DEFAULT 0,
    processados INTEGER DEFAULT 0,
    erros INTEGER DEFAULT 0,
    relatorio_path TEXT,
    manifest_json TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS importacao_item (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lote_id INTEGER NOT NULL REFERENCES importacao_lote(id),
    caminho_original TEXT NOT NULL,
    sha256 TEXT,
    pasta_origem TEXT,
    tipo_documento TEXT,
    folha INTEGER,
    face TEXT,
    termo_inicial INTEGER,
    termo_final INTEGER,
    orientacao INTEGER DEFAULT 0,
    orientacao_confianca REAL DEFAULT 0,
    orientacao_metodo TEXT,
    layout_id TEXT,
    status TEXT NOT NULL DEFAULT 'pendente',
    erro TEXT,
    imagem_id INTEGER REFERENCES imagem(id),
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(lote_id, caminho_original)
);

CREATE INDEX IF NOT EXISTS idx_imagem_livro ON imagem(livro_id);
CREATE INDEX IF NOT EXISTS idx_imagem_hash ON imagem(hash_perceptual);
CREATE INDEX IF NOT EXISTS idx_imagem_status ON imagem(status);
CREATE INDEX IF NOT EXISTS idx_livro_oficio ON livro(oficio_id);
CREATE INDEX IF NOT EXISTS idx_livro_status ON livro(status);
CREATE INDEX IF NOT EXISTS idx_revisao_resolvida ON revisao(resolvida);
CREATE INDEX IF NOT EXISTS idx_ocorrencia_livro ON ocorrencia(livro_id);
CREATE INDEX IF NOT EXISTS idx_registro_livro_termo ON registro(livro_id, termo);
CREATE INDEX IF NOT EXISTS idx_registro_imagem ON registro(imagem_id);
CREATE INDEX IF NOT EXISTS idx_ocr_execucao_imagem ON ocr_execucao(imagem_id, ativo);
CREATE INDEX IF NOT EXISTS idx_ocr_deteccao_registro ON ocr_deteccao(registro_id, ativo);
CREATE INDEX IF NOT EXISTS idx_ocr_deteccao_tipo ON ocr_deteccao(tipo, valor_normalizado);
CREATE INDEX IF NOT EXISTS idx_processamento_lote_livro ON processamento_lote(livro_id, status);
CREATE INDEX IF NOT EXISTS idx_processamento_item_fila ON processamento_item(lote_id, etapa, status, id);
CREATE INDEX IF NOT EXISTS idx_processamento_item_registro ON processamento_item(registro_id, etapa);
CREATE INDEX IF NOT EXISTS idx_importacao_item_fila ON importacao_item(lote_id, status, id);
CREATE INDEX IF NOT EXISTS idx_imagem_livro_sha256 ON imagem(livro_id, sha256);
"""

SEEDS = """
INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (1, 'Acervo legado', 'Cadastro anterior; mantido apenas por compatibilidade', datetime('now'));

INSERT OR IGNORE INTO oficio (id, nome, ativo) VALUES (6, '6º Ofício', 1);
INSERT OR IGNORE INTO oficio (id, nome, ativo) VALUES (9, '9º Ofício (atual)', 1);
INSERT OR IGNORE INTO oficio (id, nome, ativo) VALUES (12, '12º Ofício', 1);
INSERT OR IGNORE INTO oficio (id, nome, ativo) VALUES (13, '13º Ofício', 1);
INSERT OR IGNORE INTO oficio (id, nome, ativo) VALUES (14, '14º Ofício', 1);
INSERT OR IGNORE INTO oficio (id, nome, ativo) VALUES (15, '15º Ofício', 1);

UPDATE oficio SET nome='6º Ofício', ativo=1 WHERE id=6;
UPDATE oficio SET nome='9º Ofício (atual)', ativo=1 WHERE id=9;
UPDATE oficio SET nome='12º Ofício', ativo=1 WHERE id=12;
UPDATE oficio SET nome='13º Ofício', ativo=1 WHERE id=13;
UPDATE oficio SET nome='14º Ofício', ativo=1 WHERE id=14;
UPDATE oficio SET nome='15º Ofício', ativo=1 WHERE id=15;
UPDATE oficio SET ativo=0 WHERE id IN (1, 2, 3, 4, 5);

INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (6, 'Acervo do 6º Ofício', 'Acervo histórico atualmente sob guarda do 9º Ofício', datetime('now'));
INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (9, 'Acervo do 9º Ofício (atual)', 'Livros do ofício atual', datetime('now'));
INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (12, 'Acervo do 12º Ofício', 'Acervo histórico atualmente sob guarda do 9º Ofício', datetime('now'));
INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (13, 'Acervo do 13º Ofício', 'Acervo histórico atualmente sob guarda do 9º Ofício', datetime('now'));
INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (14, 'Acervo do 14º Ofício', 'Acervo histórico atualmente sob guarda do 9º Ofício', datetime('now'));
INSERT OR IGNORE INTO acervo (id, nome, descricao, created_at)
VALUES (15, 'Acervo do 15º Ofício', 'Acervo histórico atualmente sob guarda do 9º Ofício', datetime('now'));
UPDATE acervo SET ativo=0 WHERE id=1;

INSERT OR IGNORE INTO tipo_registro (id, nome, subtipo_de) VALUES (1, 'Nascimento', NULL);
INSERT OR IGNORE INTO tipo_registro (id, nome, subtipo_de) VALUES (2, 'Casamento', NULL);
INSERT OR IGNORE INTO tipo_registro (id, nome, subtipo_de) VALUES (3, 'Óbito', NULL);
INSERT OR IGNORE INTO tipo_registro (id, nome, subtipo_de) VALUES (4, 'Civil', 2);
INSERT OR IGNORE INTO tipo_registro (id, nome, subtipo_de) VALUES (5, 'Religioso com efeito civil', 2);

INSERT OR IGNORE INTO sessao (id) VALUES (1);
"""


class Database:
    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = data_dir() / "digitalizador.db"
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        existed = self._db_path.is_file()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL mantem a atomicidade do WAL e evita um fsync completo a cada
        # atualizacao de progresso. No disco operacional isso reduz segundos
        # de espera por item para milissegundos.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        revision = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if revision < SCHEMA_REVISION:
            if existed and self._needs_schema_backup():
                self._backup_before_migration()
            self._conn.executescript(f"BEGIN IMMEDIATE;\n{SCHEMA}\nCOMMIT;")
            self._conn.execute("BEGIN IMMEDIATE")
            self._migrate()
            self._conn.commit()
            self._conn.executescript(
                f"BEGIN IMMEDIATE;\n{SEEDS}\n"
                f"PRAGMA user_version={SCHEMA_REVISION};\nCOMMIT;"
            )
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._db_path

    def _needs_schema_backup(self) -> bool:
        tables = {
            row[0] for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "imagem" not in tables:
            return False
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(imagem)").fetchall()
        }
        book_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(livro)").fetchall()
        }
        return (
            "importacao_lote" not in tables
            or "caminho_normalizado" not in columns
            or "layout_id" not in book_columns
        )

    def _backup_before_migration(self) -> None:
        backup_dir = self._db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = backup_dir / f"{self._db_path.stem}.pre-v{APP_VERSION}-{stamp}.db"
        destination = sqlite3.connect(str(target))
        try:
            self._conn.backup(destination)
        finally:
            destination.close()

    def _migrate(self) -> None:
        """Aplica migracoes pequenas sem apagar bancos ja existentes."""
        colunas_imagem = {
            row[1] for row in self._conn.execute("PRAGMA table_info(imagem)").fetchall()
        }
        if "dhash" not in colunas_imagem:
            self._conn.execute("ALTER TABLE imagem ADD COLUMN dhash TEXT")
        if "qualidade_oclusao" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN qualidade_oclusao REAL DEFAULT 0"
            )
        if "qualidade_motivos" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN qualidade_motivos TEXT"
            )
        if "tipo_documento" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN tipo_documento TEXT DEFAULT 'registro'"
            )
        if "rotacao_visualizacao" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN rotacao_visualizacao INTEGER DEFAULT 0"
            )
        if "origem_posicao" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN origem_posicao INTEGER"
            )
        if "caminho_armazenamento" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN caminho_armazenamento TEXT"
            )
        if "sha256_armazenamento" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN sha256_armazenamento TEXT"
            )
        for coluna, definicao in (
            ("caminho_normalizado", "TEXT"),
            ("sha256_normalizado", "TEXT"),
            ("orientacao_confianca", "REAL"),
            ("orientacao_metodo", "TEXT"),
            ("orientacao_motivo", "TEXT"),
            ("normalizacao_json", "TEXT"),
        ):
            if coluna not in colunas_imagem:
                self._conn.execute(
                    f"ALTER TABLE imagem ADD COLUMN {coluna} {definicao}"
                )
        if "registros_detectados" not in colunas_imagem:
            self._conn.execute(
                "ALTER TABLE imagem ADD COLUMN registros_detectados INTEGER"
            )
        if "layout_id" not in colunas_imagem:
            self._conn.execute("ALTER TABLE imagem ADD COLUMN layout_id TEXT")
        if "layout_confidence" not in colunas_imagem:
            self._conn.execute("ALTER TABLE imagem ADD COLUMN layout_confidence REAL")
        if "layout_method" not in colunas_imagem:
            self._conn.execute("ALTER TABLE imagem ADD COLUMN layout_method TEXT")
        if "layout_reason" not in colunas_imagem:
            self._conn.execute("ALTER TABLE imagem ADD COLUMN layout_reason TEXT")
        colunas_livro = {
            row[1] for row in self._conn.execute("PRAGMA table_info(livro)").fetchall()
        }
        if "acervo_id" not in colunas_livro:
            self._conn.execute(
                "ALTER TABLE livro ADD COLUMN acervo_id INTEGER DEFAULT 1 REFERENCES acervo(id)"
            )
        for coluna, definicao in (
            ("registros_detectados", "INTEGER"),
            ("layout_id", "TEXT"),
            ("layout_confidence", "REAL"),
            ("layout_method", "TEXT"),
            ("layout_reason", "TEXT"),
            ("conferido_em", "TEXT"),
        ):
            if coluna not in colunas_livro:
                self._conn.execute(f"ALTER TABLE livro ADD COLUMN {coluna} {definicao}")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_livro_acervo ON livro(acervo_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_imagem_livro_sha256 "
            "ON imagem(livro_id, sha256)"
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure(self) -> None:
        if self._conn is None:
            self.connect()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            self._ensure()
            return self._conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            self._ensure()
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            self._ensure()
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def insert(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            self._ensure()
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    def update(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._ensure()
            self._conn.execute(sql, params)
            self._conn.commit()

    def executemany(self, sql: str, params: list[tuple]) -> None:
        if not params:
            return
        with self._lock:
            self._ensure()
            self._conn.executemany(sql, params)
            self._conn.commit()

    def commit(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.commit()
