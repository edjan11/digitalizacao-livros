"""Testes basicos do Digitalizador de Livros."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestDatabase:
    def test_connect(self, tmp_path):
        from src.database.connection import Database
        db = Database(tmp_path / "test.db")
        db.connect()
        assert db._conn is not None
        db.close()

    def test_oficios_seed(self, tmp_path):
        from src.database.connection import Database
        from src.database.repository import Repository
        db = Database(tmp_path / "test.db")
        db.connect()
        repo = Repository(db)
        oficios = repo.listar_oficios()
        assert len(oficios) == 6
        assert [item["id"] for item in oficios] == [6, 9, 12, 13, 14, 15]
        assert oficios[0]["nome"] == "6º Ofício"
        db.close()

    def test_tipos_seed(self, tmp_path):
        from src.database.connection import Database
        from src.database.repository import Repository
        db = Database(tmp_path / "test.db")
        db.connect()
        repo = Repository(db)
        tipos = repo.listar_tipos()
        assert len(tipos) == 3
        names = {t["nome"] for t in tipos}
        assert names == {"Nascimento", "Casamento", "Óbito"}
        subtipos = repo.listar_tipos(2)
        assert len(subtipos) == 2
        db.close()

    def test_criar_livro(self, tmp_path):
        from src.database.connection import Database
        from src.database.repository import Repository
        db = Database(tmp_path / "test.db")
        db.connect()
        repo = Repository(db)
        livro_id = repo.criar_livro(
            oficio_id=6, tipo_id=1, codigo="A-12", nome_capa="Teste",
            total_folhas=300, primeira_folha=1, ultima_folha=300,
            frente_verso=1, registros_por_face=2,
            termo_inicial=1, termo_final=1200,
        )
        assert livro_id > 0
        livro = repo.get_livro(livro_id)
        assert livro["codigo"] == "A-12"
        assert livro["status"] == "em_andamento"
        db.close()

    def test_sessao(self, tmp_path):
        from src.database.connection import Database
        from src.database.repository import Repository
        from src.session.scan_session import ScanSession
        db = Database(tmp_path / "test.db")
        db.connect()
        repo = Repository(db)
        session = ScanSession(repo)
        assert not session.tem_sessao_ativa()
        session.selecionar_oficio(6)
        session.selecionar_tipo(1)
        assert session.oficio_id == 6
        assert session.tipo_id == 1
        session.limpar_sessao()
        assert session.oficio_id is None
        db.close()


class TestImaging:
    def test_blur_detection(self):
        import numpy as np
        from src.imaging.quality import detectar_foco
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        valor, status = detectar_foco(img)
        assert isinstance(valor, float)

    def test_exposure_detection(self):
        import numpy as np
        from src.imaging.quality import detectar_exposicao
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        valor, status = detectar_exposicao(img)
        assert status == "ok"

    def test_thumbnail(self):
        import numpy as np
        from src.imaging.thumbnail import gerar_thumbnail
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        thumb = gerar_thumbnail(img, 200)
        assert thumb.shape[1] == 200


class TestHashing:
    def test_phash(self):
        import numpy as np
        from src.duplicate.hashing import compute_phash, hash_distance
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) > 0
        assert hash_distance(h, h) == 0

    def test_hash_different_images(self):
        import numpy as np
        from src.duplicate.hashing import compute_phash, hash_distance
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.ones((200, 200, 3), dtype=np.uint8) * 255
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        assert hash_distance(h1, h2) > 0
