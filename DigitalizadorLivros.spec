# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

shared_root = Path(SPECPATH).resolve().parent / 'componentes_compartilhados'

a = Analysis(
    ['app.py'],
    pathex=[str(shared_root)] if shared_root.exists() else [],
    binaries=[],
    datas=[('src', 'src')],
    hiddenimports=[
        'skimage.metrics',
        'imagehash',
        'watchdog.observers',
        'watchdog.events',
        'rapidocr_onnxruntime',
        'rapidfuzz',
        'unidecode',
    ],
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
    name='DigitalizadorLivros',
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
    name='DigitalizadorLivros',
)
