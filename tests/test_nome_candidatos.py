from __future__ import annotations

from pathlib import Path
import sys

from src.ocr.base import OCRResult
from src.ocr.name_candidates import NameCandidateIndexer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_consulta import _acervo_com_imagem
from src.session.scan_session import ScanSession


class _Provider:
    name = "tesseract"

    def __init__(self, texto: str):
        self.texto = texto

    def is_available(self):
        return True

    def recognize(self, _image, fast=False):
        return OCRResult(motor=self.name, texto_bruto=self.texto, tempo_ms=3)


def test_nome_rapido_fica_pesquisavel_e_incerto_vai_para_qwen(tmp_path):
    db, repo, livro_id, imagem_id, registros = _acervo_com_imagem(tmp_path)
    resultado = NameCandidateIndexer(
        repo,
        [_Provider("que recebeu o nome de Ana Beatriz de Souza")],
    ).indexar(imagem_id, __import__("numpy").zeros((600, 400, 3), dtype="uint8"), registros)

    assert len(resultado["nomes"]) == 2
    assert len(resultado["incertos"]) == 2
    deteccoes = db.fetchall(
        "SELECT tipo, fonte, status, confianca FROM ocr_deteccao WHERE imagem_id=? AND ativo=1",
        (imagem_id,),
    )
    assert len([d for d in deteccoes if d["fonte"] == "ocr_nome_rapido"]) == 2
    assert repo.tem_revisao_pendente(imagem_id, "nome_incerto")
    db.close()
