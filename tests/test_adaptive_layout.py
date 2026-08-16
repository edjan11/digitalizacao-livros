from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.imaging.adaptive_layout import AdaptiveLayoutDetector
from src.imaging.record_regions import bbox_numero_termo
from src.database.connection import Database
from src.database.repository import Repository
from src.services.scan_pipeline import ScanPipeline
from src.session.scan_session import ScanSession


def _pagina(dois_registros: bool) -> np.ndarray:
    imagem = np.full((800, 600, 3), 245, dtype=np.uint8)
    for y in range(70, 760, 38):
        if dois_registros or not 300 <= y <= 500:
            x2 = 560 if dois_registros else 280
            cv2.line(imagem, (40, y), (x2, y), (80, 80, 80), 2)
    cv2.line(imagem, (80, 20), (80, 780), (60, 60, 60), 3)
    cv2.line(imagem, (450, 20), (450, 780), (60, 60, 60), 3)
    if dois_registros:
        cv2.line(imagem, (35, 400), (565, 400), (30, 30, 30), 8)
    return imagem


def test_detector_separa_um_e_dois_registros():
    dois = AdaptiveLayoutDetector.observar(_pagina(True))
    um = AdaptiveLayoutDetector.observar(_pagina(False))

    assert dois.records_per_face == 2
    assert dois.separator_y is not None
    assert dois.confidence >= 0.72
    assert um.records_per_face == 1
    assert um.separator_y is None
    assert len(dois.name_bboxes) == 2
    assert len(um.term_bboxes) == 1


def test_caixa_de_termo_e_normalizada():
    bbox = bbox_numero_termo(1, 2)
    assert all(0 <= value <= 1 for value in bbox)
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]


def test_store_cria_candidato_e_persiste_mudanca_de_layout(tmp_path: Path):
    detector = AdaptiveLayoutDetector(tmp_path / "layouts.json")
    primeira = detector.classificar(_pagina(True), page_number=1, expected_records=2)
    segunda = detector.classificar(_pagina(True), page_number=2, expected_records=2)
    diferente = detector.classificar(_pagina(False), page_number=3, expected_records=2)

    assert primeira.layout_id == "layout_001"
    assert primeira.needs_review is True
    assert segunda.layout_id == "layout_001"
    assert diferente.records_per_face == 1
    assert diferente.needs_review is True
    assert (tmp_path / "layouts.json").exists()
    assert len(detector.store.templates) == 2


def test_pipeline_usa_um_assento_quando_layout_confirma(tmp_path: Path, monkeypatch):
    db = Database(tmp_path / "teste.db")
    db.connect()
    repo = Repository(db)
    livro_id = repo.criar_livro(
        oficio_id=6,
        tipo_id=1,
        codigo="A-07",
        total_folhas=300,
        primeira_folha=1,
        ultima_folha=300,
        frente_verso=1,
        registros_por_face=2,
        termo_inicial=6801,
        termo_final=8000,
    )
    session = ScanSession(repo)
    session.selecionar_livro(livro_id)
    monkeypatch.setattr(ScanPipeline, "_init_ocr", lambda self: None)
    pipeline = ScanPipeline(repo, session, tmp_path / "acervo")

    caminho = tmp_path / "face_um_assento.png"
    assert cv2.imwrite(str(caminho), _pagina(False))
    resultado = pipeline.processar_imagem_imediato(str(caminho))

    assert resultado["tipo_documento"] == "registro"
    assert resultado["registros_detectados"] == 1
    assert (resultado["termo_inicial"], resultado["termo_final"]) == (6801, 6801)
    assert session.intervalo_termos_atual == (6802, 6803)
    registros = repo.listar_registros_imagem(resultado["imagem_id"])
    assert len(registros) == 1
    armazenada = Path(resultado["caminho_armazenamento"])
    assert armazenada.is_file()
    with Image.open(armazenada) as imagem_armazenada:
        assert tuple(round(valor) for valor in imagem_armazenada.info["dpi"]) == (300, 300)
