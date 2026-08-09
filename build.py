import os
import subprocess
import shutil
import glob
import customtkinter

import sign

# Spec versionado que construye los tres ejecutables compartiendo un unico
# runtime. Ver la cabecera del propio fichero para el porque.
BUNDLE_SPEC = "ghost_bundle.spec"
BUNDLE_DIR = os.path.join("dist", "GhostTerminal")


def folder_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / (1024 * 1024)


def build():
    print("[*] Iniciando proceso de empaquetado del Ghost Terminal...")

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    if os.path.exists("build"): shutil.rmtree("build")
    if os.path.exists("dist"): shutil.rmtree("dist")

    # Eliminar spec files GENERADOS por compilaciones anteriores. ghost_bundle.spec
    # NO se toca: es codigo fuente versionado, no un artefacto.
    for spec_file in glob.glob("*.spec"):
        if os.path.basename(spec_file) != BUNDLE_SPEC:
            os.remove(spec_file)

    # Ruta absoluta de customtkinter para forzar su empaquetado
    ctk_path = os.path.dirname(customtkinter.__file__)
    print(f"[*] CustomTkinter encontrado en: {ctk_path}")

    print("\n[*] Compilando el paquete unico (terminal + worker + desinstalador)...")
    print("    Los tres comparten un solo runtime: PyTorch se incluye una vez.")
    subprocess.run(["pyinstaller", "--noconfirm", "--clean", BUNDLE_SPEC], check=True)

    if not os.path.isdir(BUNDLE_DIR):
        raise SystemExit(f"[ERROR] No se genero {BUNDLE_DIR}; revisa la salida de PyInstaller.")
    print(f"[+] Paquete compartido: {folder_size_mb(BUNDLE_DIR):.0f} MB en {BUNDLE_DIR}")

    # Se firman los ejecutables ANTES de empaquetar el setup, porque el setup los
    # embebe y los extrae ya firmados en la maquina del usuario.
    # Si no hay certificado configurado, sign_many avisa y sigue sin firmar.
    print("\n[*] Firmando binarios internos (ghost_terminal, p2p_node, uninstaller)...")
    inner_binaries = [
        os.path.join(BUNDLE_DIR, "ghost_terminal.exe"),
        os.path.join(BUNDLE_DIR, "p2p_node.exe"),
        os.path.join(BUNDLE_DIR, "uninstaller.exe"),
    ]
    signed = sign.sign_many(inner_binaries)

    print("\n[*] Construyendo el instalador mágico (setup_wizard.py)...")
    # El instalador NO embebe el paquete: se distribuye a su lado. Embeberlo
    # obligaria al usuario a tener el doble de espacio libre (extraer el .zip y
    # ademas copiar la instalacion) y duplicaria 558 MB dentro del propio .exe.
    subprocess.run([
        "pyinstaller", "--noconfirm", "--onedir", "--windowed", "--clean", "--uac-admin",
        "--name", "Ghost Terminal Setup",
        "--icon", "ghost.ico",
        "--hidden-import", "customtkinter",
        "--add-data", f"{ctk_path};customtkinter/",
        "--add-data", f"src/setup_bg.jpg;.",
        "--add-data", f"src/config.py;src",
        "src/setup_wizard.py"
    ], check=True)

    # El setup se firma el ULTIMO: es el .exe que el usuario ejecuta primero y
    # el que Smart App Control evalua al arrancar la instalacion.
    setup_dir = os.path.join("dist", "Ghost Terminal Setup")
    setup_exe = os.path.join(setup_dir, "Ghost Terminal Setup.exe")
    if signed:
        print("\n[*] Firmando el instalador...")
        sign.sign_many([setup_exe])

    release_dir = assemble_release(setup_dir)

    print("\n[+] Construcción finalizada.")
    print(f"    Carpeta a comprimir y publicar: {release_dir} "
          f"({folder_size_mb(release_dir):.0f} MB sin comprimir)")
    if not signed:
        print("[!] AVISO: los binarios NO estan firmados. Smart App Control / "
              "SmartScreen los bloquearan en otras maquinas. Ver docs/firma-codigo.md.")


def assemble_release(setup_dir):
    """Monta la carpeta que se comprime y se sube a la release de GitHub.

    Queda el instalador junto al paquete de la aplicacion, no dentro de el:
    `setup_wizard.find_resource` busca 'GhostTerminal' al lado del propio .exe.
    """
    release_dir = os.path.join("dist", "release")
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)

    shutil.copytree(setup_dir, release_dir, dirs_exist_ok=True)
    shutil.copytree(BUNDLE_DIR, os.path.join(release_dir, "GhostTerminal"))
    return release_dir

if __name__ == "__main__":
    build()
