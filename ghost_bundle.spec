# -*- mode: python ; coding: utf-8 -*-
"""Paquete unico con los tres ejecutables compartiendo un solo runtime.

Antes cada binario se construia con --onefile por separado, y tanto
ghost_terminal.exe como p2p_node.exe embebian su PROPIA copia de PyTorch
(271 MB + 273 MB). El instalador acababa pesando 602 MB de los cuales ~270 MB
eran duplicado puro.

Aqui las tres aplicaciones se analizan por separado pero se vuelcan en un unico
COLLECT: PyInstaller deduplica por ruta de destino, asi que torch_cpu.dll (305 MB
sin comprimir) aparece una sola vez en _internal/.

Ademas se abandona --onefile, que descomprimia todo el bundle en %TEMP% EN CADA
ARRANQUE (de ahi la lentitud al abrir) y cuya autoextraccion es una de las
heuristicas clasicas de falso positivo de los antivirus.
"""

import os

from PyInstaller.utils.hooks import collect_all

import customtkinter

CTK_PATH = os.path.dirname(customtkinter.__file__)

# Los modulos de src se anaden como datos ademas de analizarse: los scripts
# manipulan sys.path en tiempo de ejecucion y el analizador estatico no siempre
# sigue los imports 'from src.x import y'.
SRC_DATAS = [
    ('src/chat_agent.py', 'src'),
    ('src/tensor_utils.py', 'src'),
    ('src/p2p_node.py', 'src'),
    ('src/routing.py', 'src'),
    ('src/pow_utils.py', 'src'),
    ('src/tls_utils.py', 'src'),
    ('src/config.py', 'src'),
    ('src/ratelimit.py', 'src'),
    ('src/worker.py', 'src'),
]


def collected(*packages):
    """Equivalente en spec de --collect-all, acumulando varios paquetes."""
    datas, binaries, hiddenimports = [], [], []
    for package in packages:
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden
    return datas, binaries, hiddenimports


# --- Ghost Terminal (interfaz) ------------------------------------------------
terminal_datas, terminal_binaries, terminal_hidden = collected(
    'keyboard', 'msgpack', 'cryptography', 'pystray', 'PIL'
)

a_terminal = Analysis(
    ['src/ghost_terminal.py'],
    pathex=[],
    binaries=terminal_binaries,
    datas=terminal_datas + SRC_DATAS + [(CTK_PATH, 'customtkinter/')],
    hiddenimports=terminal_hidden + ['customtkinter', 'torch', 'transformers', 'numpy'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    # noarchive=True: el bytecode sale del PYZ del .exe a .pyc sueltos en
    # _internal, con lo que COLLECT tambien lo deduplica entre los tres
    # ejecutables. Medido sobre el paquete completo:
    #   noarchive=False -> 361 MB de descarga, 826 MB instalados,  7.201 ficheros
    #   noarchive=True  -> 286 MB de descarga, 875 MB instalados, 17.020 ficheros
    # Se acepta el coste (+49 MB en disco y mas ficheros que copiar) a cambio de
    # un 21% menos de descarga, que es la barrera real de entrada.
    noarchive=True,
)

# --- Worker P2P ---------------------------------------------------------------
# El binario del worker sale de worker.py, no de p2p_node.py: este ultimo solo
# define la clase P2PNode y no tiene __main__, asi que producia un .exe inerte.
worker_datas, worker_binaries, worker_hidden = collected(
    'msgpack', 'transformers', 'cryptography'
)

a_worker = Analysis(
    ['src/worker.py'],
    pathex=[],
    binaries=worker_binaries,
    datas=worker_datas + SRC_DATAS,
    hiddenimports=worker_hidden + ['torch', 'numpy', 'transformers'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    # noarchive=True: el bytecode sale del PYZ del .exe a .pyc sueltos en
    # _internal, con lo que COLLECT tambien lo deduplica entre los tres
    # ejecutables. Medido sobre el paquete completo:
    #   noarchive=False -> 361 MB de descarga, 826 MB instalados,  7.201 ficheros
    #   noarchive=True  -> 286 MB de descarga, 875 MB instalados, 17.020 ficheros
    # Se acepta el coste (+49 MB en disco y mas ficheros que copiar) a cambio de
    # un 21% menos de descarga, que es la barrera real de entrada.
    noarchive=True,
)

# --- Desinstalador ------------------------------------------------------------
# No necesita torch: pesa unos 9 MB por su cuenta.
a_uninstaller = Analysis(
    ['src/uninstaller.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['psutil'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    # noarchive=True: el bytecode sale del PYZ del .exe a .pyc sueltos en
    # _internal, con lo que COLLECT tambien lo deduplica entre los tres
    # ejecutables. Medido sobre el paquete completo:
    #   noarchive=False -> 361 MB de descarga, 826 MB instalados,  7.201 ficheros
    #   noarchive=True  -> 286 MB de descarga, 875 MB instalados, 17.020 ficheros
    # Se acepta el coste (+49 MB en disco y mas ficheros que copiar) a cambio de
    # un 21% menos de descarga, que es la barrera real de entrada.
    noarchive=True,
)

pyz_terminal = PYZ(a_terminal.pure, a_terminal.zipped_data)
pyz_worker = PYZ(a_worker.pure, a_worker.zipped_data)
pyz_uninstaller = PYZ(a_uninstaller.pure, a_uninstaller.zipped_data)

# exclude_binaries=True: las dependencias no van dentro del .exe, sino a la
# carpeta compartida que arma COLLECT.
exe_terminal = EXE(
    pyz_terminal, a_terminal.scripts, [],
    exclude_binaries=True,
    name='ghost_terminal',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False, icon='ghost.ico',
)

exe_worker = EXE(
    pyz_worker, a_worker.scripts, [],
    exclude_binaries=True,
    name='p2p_node',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False,
)

exe_uninstaller = EXE(
    pyz_uninstaller, a_uninstaller.scripts, [],
    exclude_binaries=True,
    name='uninstaller',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False,
)

# Un unico COLLECT con los tres ejecutables: las dependencias comunes se
# escriben una sola vez porque comparten ruta de destino.
coll = COLLECT(
    exe_terminal, a_terminal.binaries, a_terminal.datas,
    exe_worker, a_worker.binaries, a_worker.datas,
    exe_uninstaller, a_uninstaller.binaries, a_uninstaller.datas,
    strip=False, upx=True, upx_exclude=[],
    name='GhostTerminal',
)
