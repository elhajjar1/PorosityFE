# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Porosity FE Analysis Mac App

import sys
import os

block_cipher = None

a = Analysis(
    ['porosity_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('porosity_fe_analysis.py', '.'),
    ],
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.interpolate',
        'scipy.linalg',
        'scipy.sparse',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'mpl_toolkits.mplot3d',
        'mpl_toolkits.mplot3d.art3d',
        'json',
        'dataclasses',
        'porosity_fe_analysis',
        'appdirs',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
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
        'pkg_resources',
        'tkinter',
        '_tkinter',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PorosityFE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
    name='PorosityFE',
)

app = BUNDLE(
    coll,
    name='PorosityFE.app',
    icon=None,
    bundle_identifier='com.composites.porosity-fe',
    info_plist={
        'CFBundleName': 'Porosity FE Analysis',
        'CFBundleDisplayName': 'Porosity FE Analysis',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
    },
)
