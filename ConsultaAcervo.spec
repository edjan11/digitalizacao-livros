# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

transformers_hidden = collect_submodules('transformers.models.got_ocr2')
qwen_hidden = collect_submodules('transformers.models.qwen2_vl')

a = Analysis(
    ['consulta.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src')],
    hiddenimports=[
        'skimage.metrics',
        'imagehash',
        'rapidocr_onnxruntime',
        'rapidfuzz',
        'unidecode',
        'huggingface_hub',
        'safetensors',
    ] + transformers_hidden + qwen_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ConsultaAcervo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConsultaAcervo',
)
