from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from src.database.connection import Database
from src.database.repository import Repository
from src.services.generic_book_importer import (
    A16_MISSING_FACES,
    BookImportSpec,
    GenericBookImporter,
    auditar_livro,
)


def _jpg(path: Path, value: int) -> None:
    image = np.full((240, 160, 3), 240, dtype=np.uint8)
    cv2.line(image, (15, 30), (145, 30), (value, value, value), 3)
    cv2.line(image, (20, 120), (140, 120), (20, 20, 20), 4)
    assert cv2.imwrite(str(path), image)


def _spec() -> BookImportSpec:
    return BookImportSpec(
        codigo="A-TESTE",
        acervo_id=12,
        oficio_id=12,
        tipo_id=1,
        total_folhas=5,
        termo_inicial=101,
        termo_final=120,
        registros_por_face=2,
    )


def test_auditoria_nao_desloca_termos_quando_falta_face(tmp_path: Path):
    frente = tmp_path / "frente"
    verso = tmp_path / "VERSO"
    indice = tmp_path / "INDECE"
    frente.mkdir(); verso.mkdir(); indice.mkdir()
    pages = {}
    # Faltam frente 3 e verso 4. Um indice foi colocado em frente.
    for folder, face, folhas in ((frente, "frente", (1, 2, 4, 5)), (verso, "verso", (1, 2, 3, 5))):
        for pos, folha in enumerate(folhas):
            path = folder / f"IMG_{face}_{pos:02}.jpg"
            _jpg(path, 30 + folha)
            pages[str(path)] = folha
    misplaced = frente / "IMG_indice_perdido.jpg"
    _jpg(misplaced, 99)
    (indice / "indice.jpg").write_bytes(misplaced.read_bytes())

    audit = auditar_livro(
        tmp_path,
        _spec(),
        page_resolver=lambda path, _face: pages.get(str(path)),
    )

    assert {(item.folha, item.face) for item in audit.registros} == {
        (1, "frente"), (2, "frente"), (4, "frente"), (5, "frente"),
        (1, "verso"), (2, "verso"), (3, "verso"), (5, "verso"),
    }
    face_4 = next(item for item in audit.registros if item.folha == 4 and item.face == "frente")
    assert (face_4.termo_inicial, face_4.termo_final) == (113, 114)
    assert {(item.folha, item.face) for item in audit.faltantes} == {(3, "frente"), (4, "verso")}
    assert misplaced in audit.indices


def test_importacao_idempotente_e_nao_altera_a13(tmp_path: Path):
    root = tmp_path / "livro"
    frente = root / "frente"; verso = root / "VERSO"; indice = root / "INDECE"
    frente.mkdir(parents=True); verso.mkdir(); indice.mkdir()
    pages = {}
    for folder, face in ((frente, "frente"), (verso, "verso")):
        for folha in range(1, 6):
            path = folder / f"{face}_{folha}.jpg"
            _jpg(path, 20 + folha + (40 if face == "verso" else 0))
            pages[str(path)] = folha
    audit = auditar_livro(root, _spec(), page_resolver=lambda p, _f: pages.get(str(p)))

    db = Database(tmp_path / "db.sqlite")
    db.connect()
    repo = Repository(db)
    a13 = repo.criar_livro(acervo_id=13, oficio_id=13, tipo_id=1, codigo="A-13")
    snapshot = db.fetchone("SELECT * FROM livro WHERE id=?", (a13,))

    importer = GenericBookImporter(repo, normalized_root=tmp_path / "normalizadas")
    first = importer.importar(audit, create_derivatives=False)
    second = importer.importar(audit, create_derivatives=False)

    assert first["registros"] == 20
    assert second["novas_imagens"] == 0
    assert db.fetchone("SELECT COUNT(*) n FROM imagem WHERE livro_id=?", (first["livro_id"],))["n"] == 10
    assert db.fetchone("SELECT COUNT(*) n FROM registro WHERE livro_id=?", (first["livro_id"],))["n"] == 20
    assert db.fetchone("SELECT * FROM livro WHERE id=?", (a13,)) == snapshot
    for item in audit.registros:
        assert hashlib.sha256(item.path.read_bytes()).hexdigest() == item.sha256
    db.close()


def test_manifesto_a16_preserva_lacunas_auditadas():
    assert A16_MISSING_FACES == {
        (27, "frente"), (37, "frente"), (5, "verso")
    }
    spec = BookImportSpec.a16()
    assert spec.terms(27, "frente") == (16907, 16908)
    assert spec.terms(37, "frente") == (16947, 16948)
    assert spec.terms(5, "verso") == (16821, 16822)
