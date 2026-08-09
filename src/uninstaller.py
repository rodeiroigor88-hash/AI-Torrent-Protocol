"""Desinstalador de Ghost Terminal y del worker TokenTorrent.

Orden deliberado: **primero el registro, luego los procesos, luego los ficheros**.
Si el usuario cierra el desinstalador a medias, lo peor que puede pasar es que
queden ficheros huérfanos; lo que nunca debe quedar es una entrada de
auto-arranque que resucite la aplicación en el siguiente reinicio.
"""

import os
import sys
import subprocess
import time

if os.name == "nt":
    import winreg

try:
    import psutil
except ImportError:  # el .exe deberia traerlo; si no, quedan taskkill y wmic
    psutil = None

PROCESS_NAMES = ("ghost_terminal.exe", "p2p_node.exe", "Ghost Terminal Setup.exe")
RUN_VALUES = ("GhostTerminal", "TokenTorrentWorker")
ENVIRONMENT_VALUES = (
    "TOKENTORRENT_AUTH_TOKEN", "TOKENTORRENT_TLS_CERT", "TOKENTORRENT_TLS_KEY",
    "TOKENTORRENT_TLS_CA", "TOKENTORRENT_TRACKER_KEY", "TOKENTORRENT_TRACKER_URL",
    "TOKENTORRENT_CONFIG",
)
KILL_ROUNDS = 5


def _no_window_kwargs():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def kill_process(name):
    """Mata el proceso y todo su árbol de hijos (/T)."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/IM", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       **_no_window_kwargs())
    except OSError:
        pass


def survivors(install_dir):
    """Procesos nuestros que siguen vivos, por nombre o por ruta de ejecutable.

    Comprobar también la ruta detecta la copia renombrada que un `taskkill /IM`
    no vería: sin esto quedaba un proceso zombi reteniendo los ficheros.
    """
    if psutil is None:
        return []
    alive = []
    target_dir = os.path.normcase(os.path.abspath(install_dir))
    me = os.getpid()
    for process in psutil.process_iter(["pid", "name", "exe"]):
        if process.info["pid"] == me:
            continue
        name = (process.info.get("name") or "").lower()
        executable = process.info.get("exe") or ""
        try:
            in_install_dir = bool(executable) and os.path.normcase(
                os.path.abspath(executable)).startswith(target_dir + os.sep)
        except (OSError, ValueError):
            in_install_dir = False
        if name in {n.lower() for n in PROCESS_NAMES} or in_install_dir:
            alive.append(process)
    return alive


def kill_switch(install_dir):
    """Insiste hasta que no quede nada vivo, o hasta agotar los intentos."""
    for round_number in range(KILL_ROUNDS):
        for name in PROCESS_NAMES:
            kill_process(name)

        remaining = survivors(install_dir)
        if not remaining:
            return True

        for process in remaining:
            try:
                process.kill()
            except Exception:
                pass
        if psutil is not None:
            psutil.wait_procs(remaining, timeout=1.5)
        print(f"  Reintento {round_number + 1}/{KILL_ROUNDS}: "
              f"{len(remaining)} proceso(s) resistiéndose...")
        time.sleep(0.5)

    return not survivors(install_dir)


def clean_registry(install_dir):
    """Auto-arranque y variables de entorno. Se hace lo primero, a propósito."""
    try:
        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"
        ) as key:
            for value_name in RUN_VALUES:
                try:
                    winreg.DeleteValue(key, value_name)
                except FileNotFoundError:
                    pass
    except OSError as exc:
        print(f"Error limpiando el auto-arranque: {exc}")

    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            try:
                current_path, value_type = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_path, value_type = None, winreg.REG_EXPAND_SZ
            if current_path is not None:
                entries = [
                    entry for entry in current_path.split(os.pathsep)
                    if entry and os.path.normcase(entry) != os.path.normcase(install_dir)
                ]
                winreg.SetValueEx(key, "Path", 0, value_type, os.pathsep.join(entries))
            for variable in ENVIRONMENT_VALUES:
                try:
                    winreg.DeleteValue(key, variable)
                except FileNotFoundError:
                    pass
    except OSError as exc:
        print(f"Error limpiando las variables de entorno: {exc}")

    _broadcast_environment_change()


def _broadcast_environment_change():
    """Avisa a las aplicaciones abiertas de que el entorno cambió."""
    try:
        import ctypes
        HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001A, 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST, WM_SETTINGCHANGE, 0,
            ctypes.c_wchar_p("Environment"), SMTO_ABORTIFHUNG, 3000, None,
        )
    except Exception:
        pass


def remove_shortcuts():
    desktop = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "Desktop")
    for shortcut in ("Ghost Terminal.lnk", "Uninstall Ghost Terminal.lnk"):
        try:
            os.remove(os.path.join(desktop, shortcut))
        except OSError:
            pass


def data_directory():
    """Config y log del worker, que pueden estar fuera del directorio elegido."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "GhostTerminal")


def main():
    print("Iniciando desinstalacion de Ghost Terminal y TokenTorrent Worker...")
    install_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
        else os.path.dirname(os.path.abspath(__file__))

    # 1. Registro PRIMERO: aunque se cierre el desinstalador ahora mismo, la
    #    aplicacion ya no volvera a arrancar sola.
    print("Eliminando entradas del registro y variables de entorno...")
    clean_registry(install_dir)

    # 2. Kill-switch: insistir hasta que no quede nada en memoria.
    print("Deteniendo procesos en segundo plano...")
    if not kill_switch(install_dir):
        print("  [AVISO] Algun proceso sigue vivo; se borrara al reiniciar.")

    # 3. Accesos directos.
    print("Eliminando accesos directos...")
    remove_shortcuts()

    # 4. Ficheros: el propio .exe esta en uso, asi que lo borra un script externo.
    print("Borrando archivos...")
    cleanup_script = (
        "Start-Sleep -Seconds 2;"
        "foreach ($dir in @($env:GHOST_INSTALL_DIR, $env:GHOST_DATA_DIR)) {"
        "  if ($dir -and (Test-Path -LiteralPath $dir)) {"
        "    for ($i = 0; $i -lt 5; $i++) {"
        "      Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue;"
        "      if (-not (Test-Path -LiteralPath $dir)) { break };"
        "      Start-Sleep -Seconds 2"
        "    }"
        "  }"
        "}"
    )
    env = os.environ.copy()
    env["GHOST_INSTALL_DIR"] = install_dir
    env["GHOST_DATA_DIR"] = data_directory()
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cleanup_script],
            env=env, **_no_window_kwargs(),
        )
    except OSError as exc:
        print(f"No se pudo lanzar la limpieza final: {exc}")
    sys.exit(0)

if __name__ == "__main__":
    main()
