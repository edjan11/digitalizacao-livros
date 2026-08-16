from pathlib import Path

from PIL import Image

from src.imaging.storage_derivative import criar_derivada_armazenamento
from src.services.name_processing import _abrir_imagem


def test_derivada_reduz_440_para_300_sem_alterar_original(tmp_path: Path):
    source = tmp_path / "pagina.jpg"
    output = tmp_path / "armazenamento" / "pagina.jpg"
    image = Image.new("RGB", (4400, 2200), "white")
    image.save(source, dpi=(440, 440), quality=95, subsampling=0)
    source_before = source.read_bytes()

    result = criar_derivada_armazenamento(source, output, target_dpi=300, jpeg_quality=95)

    assert result.source_size == (4400, 2200)
    assert result.output_size == (3000, 1500)
    assert result.source_dpi == (440.0, 440.0)
    assert result.output_dpi == (300.0, 300.0)
    assert result.status == "redimensionada"
    assert source.read_bytes() == source_before
    with Image.open(output) as saved:
        assert saved.size == (3000, 1500)
        assert tuple(round(value) for value in saved.info["dpi"]) == (300, 300)


def test_derivada_nao_faz_upscale_de_fonte_ja_300(tmp_path: Path):
    source = tmp_path / "pagina.jpg"
    output = tmp_path / "armazenamento" / "pagina.jpg"
    Image.new("RGB", (1200, 800), "white").save(source, dpi=(300, 300), quality=95)

    result = criar_derivada_armazenamento(source, output, target_dpi=300)

    assert result.output_size == (1200, 800)
    assert result.status == "ja_nao_acima_do_alvo"


def test_leitor_da_ia_prioriza_derivada_300_dpi(tmp_path: Path):
    source = tmp_path / "pagina.jpg"
    output = tmp_path / "armazenamento" / "pagina.jpg"
    Image.new("RGB", (4400, 2200), "white").save(source, dpi=(440, 440), quality=95)
    result = criar_derivada_armazenamento(source, output, target_dpi=300)

    image, sha = _abrir_imagem({
        "caminho_original": str(source),
        "caminho_armazenamento": str(output),
        "sha256_atual": "hash-do-original-nao-usado",
        "sha256_armazenamento": result.output_sha256,
    })

    assert image.shape[:2] == (1500, 3000)
    assert sha == result.output_sha256
