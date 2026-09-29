# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone validate_porosity CLI.

Bundles all 13 validation dataset JSONs and the schema file so the
executable can run predictions offline against the reference database.

Build:
    pyinstaller ValidatePorosity.spec

Output:
    dist/validate_porosity           (Linux/macOS executable)
    dist/validate_porosity/...       (one-dir mode on macOS)

Run:
    ./dist/validate_porosity/validate_porosity
    ./dist/validate_porosity/validate_porosity --output-dir /tmp
    ./dist/validate_porosity/validate_porosity --help
"""
import glob
import os

block_cipher = None

_spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Bundle all validation dataset JSONs under validation/datasets/
_dataset_files = [
    (path, 'validation/datasets')
    for path in glob.glob(os.path.join(_spec_dir, 'validation', 'datasets',
                                        '*.json'))
]

# Bundle the schema
_schema_files = [
    (os.path.join(_spec_dir, 'validation', 'schemas',
                  'validation_dataset_schema.json'),
     'validation/schemas'),
]

# Bundle __init__.py placeholders so imports work in frozen mode
_init_files = []
_validation_init = os.path.join(_spec_dir, 'validation', '__init__.py')
if os.path.exists(_validation_init):
    _init_files.append((_validation_init, 'validation'))

_datas = _dataset_files + _schema_files + _init_files

a = Analysis(
    ['validate_porosity_cli.py'],
    pathex=[_spec_dir],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.sparse',
        'scipy.sparse.linalg',
        'scipy.linalg',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'mpl_toolkits',
        'mpl_toolkits.mplot3d',
        'mpl_toolkits.mplot3d.art3d',
        'jsonschema',
        'jsonschema.validators',
        'porosity_fe_analysis',
        'validation',
        'validation.validate_all',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
        'tkinter',
        '_tkinter',
        'PyQt6',  # CLI doesn't need Qt
        'PyQt5',
        'PySide6',
        'PIL.ImageQt',
        'pkg_resources',  # pulled by jsonschema via setuptools; not needed
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='validate_porosity',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # CLI tool — keep console
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='validate_porosity',
)
