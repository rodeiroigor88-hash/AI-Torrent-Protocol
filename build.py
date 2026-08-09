import os
import subprocess
import shutil
import glob
import customtkinter

import sign

def build():
    print("[*] Iniciando proceso de empaquetado del Ghost Terminal...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    if os.path.exists("build"): shutil.rmtree("build")
    if os.path.exists("dist"): shutil.rmtree("dist")
    
    # Eliminar spec files viejos
    for spec_file in glob.glob("*.spec"):
        os.remove(spec_file)
        
    # Ruta absoluta de customtkinter para forzar su empaquetado
    ctk_path = os.path.dirname(customtkinter.__file__)
    print(f"[*] CustomTkinter encontrado en: {ctk_path}")
    
    print("\n[*] Compilando ghost_terminal.py...")
    subprocess.run([
        "pyinstaller", "--noconfirm", "--onefile", "--windowed", "--clean",
        "--name", "ghost_terminal", 
        "--icon", "ghost.ico",
        "--hidden-import", "customtkinter",
        "--hidden-import", "torch",
        "--hidden-import", "transformers",
        "--hidden-import", "numpy",
        "--collect-all", "keyboard",
        "--collect-all", "msgpack",
        "--collect-all", "cryptography",
        # pystray carga su backend (_win32) dinamicamente: sin collect-all,
        # el .exe arranca sin icono de bandeja.
        "--collect-all", "pystray",
        "--collect-all", "PIL",
        "--add-data", f"{ctk_path};customtkinter/",
        "--add-data", f"src/chat_agent.py;src",
        "--add-data", f"src/tensor_utils.py;src",
        "--add-data", f"src/p2p_node.py;src",
        "--add-data", f"src/routing.py;src",
        "--add-data", f"src/pow_utils.py;src",
        "--add-data", f"src/tls_utils.py;src",
        "--add-data", f"src/config.py;src",
        "--add-data", f"src/ratelimit.py;src",
        "src/ghost_terminal.py"
    ], check=True)
    
    print("\n[*] Compilando worker.py (nodo P2P con punto de entrada CLI)...")
    # NOTA: el binario del Worker se genera a partir de worker.py, no de p2p_node.py.
    # p2p_node.py solo define la clase P2PNode; no tiene __main__/argparse propio,
    # así que compilarlo directamente producía un .exe que no hacía nada al ejecutarse.
    subprocess.run([
        "pyinstaller", "--noconfirm", "--onefile", "--noconsole", "--clean",
        "--name", "p2p_node",
        "--hidden-import", "torch",
        "--hidden-import", "numpy",
        "--hidden-import", "transformers",
        "--collect-all", "msgpack",
        "--collect-all", "transformers",
        "--collect-all", "cryptography",
        "--add-data", f"src/p2p_node.py;src",
        "--add-data", f"src/tensor_utils.py;src",
        "--add-data", f"src/routing.py;src",
        "--add-data", f"src/pow_utils.py;src",
        "--add-data", f"src/tls_utils.py;src",
        "--add-data", f"src/config.py;src",
        "--add-data", f"src/ratelimit.py;src",
        "src/worker.py"
    ], check=True)
    
    print("\n[*] Compilando uninstaller.py...")
    subprocess.run([
        "pyinstaller", "--noconfirm", "--onefile", "--noconsole", "--clean",
        "--name", "uninstaller",
        "--hidden-import", "psutil",
        "src/uninstaller.py"
    ], check=True)
    
    # Se firman los tres binarios internos ANTES de empaquetar el setup, porque
    # el setup los embebe y los extrae ya firmados en la maquina del usuario.
    # Si no hay certificado configurado, sign_many avisa y sigue sin firmar.
    print("\n[*] Firmando binarios internos (ghost_terminal, p2p_node, uninstaller)...")
    inner_binaries = [
        os.path.join("dist", "ghost_terminal.exe"),
        os.path.join("dist", "p2p_node.exe"),
        os.path.join("dist", "uninstaller.exe"),
    ]
    signed = sign.sign_many(inner_binaries)

    print("\n[*] Construyendo el instalador mágico (setup_wizard.py)...")
    subprocess.run([
        "pyinstaller", "--noconfirm", "--onedir", "--windowed", "--clean", "--uac-admin",
        "--name", "Ghost Terminal Setup",
        "--icon", "ghost.ico", 
        "--hidden-import", "customtkinter",
        "--add-data", f"{ctk_path};customtkinter/",
        "--add-data", f"src/setup_bg.jpg;.",
        "--add-data", f"src/config.py;src",
        "--add-data", f"dist/ghost_terminal.exe;.", 
        "--add-data", f"dist/p2p_node.exe;.",
        "--add-data", f"dist/uninstaller.exe;.",
        "src/setup_wizard.py"
    ], check=True)

    # El setup se firma el ULTIMO: es el .exe que el usuario ejecuta primero y
    # el que Smart App Control evalua al arrancar la instalacion.
    setup_exe = os.path.join("dist", "Ghost Terminal Setup", "Ghost Terminal Setup.exe")
    if signed:
        print("\n[*] Firmando el instalador...")
        sign.sign_many([setup_exe])

    print("\n[+] Construcción finalizada. Revisa la carpeta 'dist/'.")
    if not signed:
        print("[!] AVISO: los binarios NO estan firmados. Smart App Control / "
              "SmartScreen los bloquearan en otras maquinas. Ver docs/firma-codigo.md.")

if __name__ == "__main__":
    build()
