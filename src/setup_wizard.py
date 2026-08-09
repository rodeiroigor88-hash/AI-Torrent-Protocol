import customtkinter as ctk
import os
import socket
import sys
import time
import shutil
import subprocess
import threading
import secrets
import queue
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config, save_config

if os.name == "nt":
    import winreg

# Configuraciones Cypherpunk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Tamaño mínimo por debajo del cual los controles del panel inferior se solapan.
MIN_WIDTH, MIN_HEIGHT = 760, 470
GRIP = 6           # grosor en píxeles de los bordes activos de redimensionado
BG_REDRAW_MS = 60  # anti-rebote del reescalado del fondo


def find_free_port(port: int, attempts: int = 20) -> int:
    """Primer puerto libre a partir de `port`.

    Si otra aplicación ya ocupa el 8001, el worker moría al arrancar sin que el
    usuario viera nada: mejor detectarlo durante la instalación.
    """
    for offset in range(max(1, attempts)):
        candidate = port + offset
        if candidate > 65535:
            break
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Sin SO_REUSEADDR: en Windows permitiría atar un puerto ya ocupado.
            probe.bind(("0.0.0.0", candidate))
            return candidate
        except OSError:
            continue
        finally:
            probe.close()
    return port

class SetupWizard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Ghost Terminal Setup")
        
        # Modo sin bordes
        self.overrideredirect(True)
        self.configure(fg_color="#000000")
        
        # Tamaño cinematográfico 16:9
        width, height = 900, 550
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Variables
        self.install_dir = os.path.join(os.environ["LOCALAPPDATA"], "GhostTerminal")
        self.worker_var = ctk.BooleanVar(value=False)
        self.path_var = ctk.BooleanVar(value=True)
        self.terms_var = ctk.BooleanVar(value=False)
        self.progress_updates = queue.Queue()
        self.settings = load_config()
        self.worker_port = self.settings["worker_port"]
        self._resize_state = None
        self._bg_redraw_job = None

        self.build_ui()
        self.build_resize_grips()
        self.bind("<Configure>", self._on_configure)
        self.after(100, self._poll_progress_updates)
        
    def find_resource(self, filename):
        if getattr(sys, 'frozen', False):
            if hasattr(sys, '_MEIPASS'):
                p1 = os.path.join(sys._MEIPASS, filename)
                if os.path.exists(p1): return p1
            p2 = os.path.join(os.path.dirname(sys.executable), filename)
            if os.path.exists(p2): return p2
            p3 = os.path.join(os.path.dirname(sys.executable), "_internal", filename)
            if os.path.exists(p3): return p3
        p4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(p4): return p4
        # Ejecutando desde el codigo fuente: el paquete compilado esta en dist/,
        # asi que el asistente se puede probar sin congelarlo antes.
        p5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "dist", filename)
        if os.path.exists(p5): return p5
        raise FileNotFoundError(f"No se pudo encontrar el recurso: {filename}")

    def build_ui(self):
        # --- 1. Fondo de pantalla completo (Cinematic Art) ---
        try:
            bg_path = self.find_resource("setup_bg.jpg")
            img = Image.open(bg_path)
            self.bg_image = ctk.CTkImage(light_image=img, dark_image=img, size=(900, 550))
            self.bg_lbl = ctk.CTkLabel(self, text="", image=self.bg_image)
            self.bg_lbl.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            self.bg_lbl = ctk.CTkLabel(self, text="Fondo no encontrado", fg_color="#0A0A0A")
            self.bg_lbl.place(x=0, y=0, relwidth=1, relheight=1)
            
        # --- Barra de Título Custom ---
        self.title_bar = ctk.CTkFrame(self, height=30, corner_radius=0, fg_color="#050505", border_width=0)
        self.title_bar.place(relx=0, rely=0, relwidth=1)
        
        # Permitir arrastrar la ventana
        self.title_bar.bind("<B1-Motion>", self.move_window)
        self.title_bar.bind("<Button-1>", self.get_pos)
        
        title_lbl = ctk.CTkLabel(self.title_bar, text=" Ghost Terminal Setup", font=("Consolas", 12), text_color="#00FF41")
        title_lbl.pack(side="left", padx=10)
        
        close_btn = ctk.CTkButton(self.title_bar, text="X", width=30, height=30, corner_radius=0, fg_color="transparent", 
                                  hover_color="#FF0000", text_color="#FFFFFF", command=self.destroy)
        close_btn.pack(side="right")
        
        min_btn = ctk.CTkButton(self.title_bar, text="-", width=30, height=30, corner_radius=0, fg_color="transparent", 
                                hover_color="#333333", text_color="#FFFFFF", command=self.iconify)
        min_btn.pack(side="right")
            
        # --- 2. Panel inferior translúcido (Controles) ---
        self.bottom_panel = ctk.CTkFrame(self, fg_color="#050505", corner_radius=0, border_width=1, border_color="#00FF41")
        self.bottom_panel.place(relx=0, rely=0.6, relwidth=1, relheight=0.4)
        
        # Grid para alinear columnas dentro del panel inferior
        self.bottom_panel.grid_columnconfigure(0, weight=1)
        self.bottom_panel.grid_columnconfigure(1, weight=1)
        
        # Columna Izquierda: Opciones de instalación
        self.options_frame = ctk.CTkFrame(self.bottom_panel, fg_color="transparent")
        self.options_frame.grid(row=0, column=0, sticky="nw", padx=30, pady=20)
        
        title = ctk.CTkLabel(self.options_frame, text="Ghost Terminal // Instalación", font=("Consolas", 18, "bold"), text_color="#00FF41")
        title.pack(anchor="w", pady=(0, 10))
        
        # --- NUEVO: Selector de ruta ---
        path_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        path_frame.pack(anchor="w", pady=(0, 10), fill="x")
        
        path_lbl = ctk.CTkLabel(path_frame, text="Ruta:", font=("Consolas", 12), text_color="#AAAAAA")
        path_lbl.pack(side="left")
        
        self.path_entry = ctk.CTkEntry(path_frame, font=("Consolas", 10), width=250, fg_color="#111111", text_color="#FFFFFF")
        self.path_entry.pack(side="left", padx=10)
        self.path_entry.insert(0, self.install_dir)
        
        def select_dir():
            d = ctk.filedialog.askdirectory(initialdir=self.install_dir)
            if d:
                self.path_entry.delete(0, "end")
                self.path_entry.insert(0, d)
                self.install_dir = d
                
        browse_btn = ctk.CTkButton(path_frame, text="Examinar...", font=("Consolas", 10), width=80,
                                   fg_color="#333333", hover_color="#555555", command=select_dir)
        browse_btn.pack(side="left")
        # -------------------------------
        
        worker_cb = ctk.CTkCheckBox(self.options_frame, text="Instalar AI Torrent Worker (Donar Recursos)", 
                                    font=("Consolas", 12), text_color="#FFFFFF", fg_color="#00FF41", 
                                    hover_color="#00CC33", variable=self.worker_var)
        worker_cb.pack(anchor="w", pady=5)
        
        path_cb = ctk.CTkCheckBox(self.options_frame, text="Añadir Ghost Terminal al PATH del sistema", 
                                  font=("Consolas", 12), text_color="#FFFFFF", fg_color="#00FF41", 
                                  hover_color="#00CC33", variable=self.path_var)
        path_cb.pack(anchor="w", pady=5)
        
        # Columna Derecha: Botón Siguiente (Página 1)
        self.action_frame_1 = ctk.CTkFrame(self.bottom_panel, fg_color="transparent")
        self.action_frame_1.grid(row=0, column=1, sticky="se", padx=30, pady=20)
        
        btn_box_1 = ctk.CTkFrame(self.action_frame_1, fg_color="transparent")
        btn_box_1.pack(anchor="e")
        
        cancel_btn_1 = ctk.CTkButton(btn_box_1, text="Abortar", font=("Consolas", 14), width=100,
                                        fg_color="#333333", text_color="#FFFFFF", hover_color="#555555", command=self.destroy)
        cancel_btn_1.pack(side="left", padx=10)
        
        next_btn = ctk.CTkButton(btn_box_1, text="Siguiente", font=("Consolas", 14, "bold"), width=120,
                                         fg_color="#005511", text_color="#00FF41", hover_color="#00CC33", command=self.show_page_2)
        next_btn.pack(side="left")

        # --- Página 2: Términos y Condiciones ---
        self.page_2_frame = ctk.CTkFrame(self.bottom_panel, fg_color="transparent")
        
        # Grid para Página 2
        self.page_2_frame.grid_columnconfigure(0, weight=1)
        
        title_2 = ctk.CTkLabel(self.page_2_frame, text="Términos y Condiciones del Enjambre P2P", font=("Consolas", 18, "bold"), text_color="#00FF41")
        title_2.pack(pady=(15, 10))
        
        # En lugar de un feo Textbox con scrollbar, usamos un frame limpio con labels
        terms_container = ctk.CTkFrame(self.page_2_frame, fg_color="#0A0A0A", corner_radius=8, border_width=1, border_color="#1A1A1A")
        terms_container.pack(padx=30, pady=5, fill="x")
        
        term1 = ctk.CTkLabel(terms_container, text="[1] DONACIÓN: Tu PC compartirá recursos inactivos con la red P2P.", 
                             font=("Consolas", 12), text_color="#CCCCCC", justify="left", anchor="w")
        term1.pack(padx=15, pady=(10, 2), fill="x")
        
        term2 = ctk.CTkLabel(terms_container, text="[2] PRIVACIDAD: El tráfico P2P está protegido por cifrado mTLS estricto.", 
                             font=("Consolas", 12), text_color="#CCCCCC", justify="left", anchor="w")
        term2.pack(padx=15, pady=2, fill="x")
        
        term3 = ctk.CTkLabel(terms_container, text="[3] EXPERIMENTAL: Sin SLA de disponibilidad. Control total para desinstalar.", 
                             font=("Consolas", 12), text_color="#CCCCCC", justify="left", anchor="w")
        term3.pack(padx=15, pady=(2, 10), fill="x")
        
        # Botones de Página 2
        self.action_frame_2 = ctk.CTkFrame(self.page_2_frame, fg_color="transparent")
        self.action_frame_2.pack(anchor="e", padx=30, pady=(10, 0))
        
        terms_cb = ctk.CTkCheckBox(self.action_frame_2, text="Acepto los Términos y Condiciones", 
                                   font=("Consolas", 12), text_color="#AAAAAA", fg_color="#00FF41", 
                                   hover_color="#00CC33", variable=self.terms_var, command=self.check_terms)
        terms_cb.pack(side="left", padx=(0, 20))
        
        back_btn = ctk.CTkButton(self.action_frame_2, text="Atrás", font=("Consolas", 14), width=100,
                                        fg_color="#333333", text_color="#FFFFFF", hover_color="#555555", command=self.show_page_1)
        back_btn.pack(side="left", padx=10)
        
        self.install_btn = ctk.CTkButton(self.action_frame_2, text="Instalar", font=("Consolas", 14, "bold"), width=120,
                                         fg_color="#005511", text_color="#00FF41", state="disabled", command=self.start_install)
        self.install_btn.pack(side="left")
        
        # Progreso (Oculto inicialmente)
        self.progress_frame = ctk.CTkFrame(self.page_2_frame, fg_color="transparent")
        self.progress = ctk.CTkProgressBar(self.progress_frame, fg_color="#1A1A1A", progress_color="#00FF41", width=250)
        self.progress.pack(pady=(0, 5))
        self.status_lbl = ctk.CTkLabel(self.progress_frame, text="Iniciando...", font=("Consolas", 11), text_color="#888888")
        self.status_lbl.pack()

    def show_page_1(self):
        self.page_2_frame.place_forget()
        self.options_frame.grid(row=0, column=0, sticky="nw", padx=30, pady=20)
        self.action_frame_1.grid(row=0, column=1, sticky="se", padx=30, pady=20)
        
    def show_page_2(self):
        self.options_frame.grid_forget()
        self.action_frame_1.grid_forget()
        self.page_2_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def check_terms(self):
        if self.terms_var.get():
            self.install_btn.configure(state="normal", fg_color="#00FF41", text_color="#000000", hover_color="#00CC33")
        else:
            self.install_btn.configure(state="disabled", fg_color="#005511", text_color="#00FF41")

    def start_install(self):
        # Transición UI
        for widget in self.action_frame_2.winfo_children():
            widget.pack_forget()
            
        self.progress_frame.pack(anchor="e", padx=20, pady=(10, 0))
        self.progress.set(0)

        selected_dir = os.path.abspath(os.path.expandvars(self.path_entry.get().strip()))
        if not selected_dir or any(ord(char) < 32 for char in selected_dir):
            self.status_lbl.configure(text="Ruta de instalación inválida")
            return
        self.install_dir = selected_dir
        self.install_worker = bool(self.worker_var.get())
        self.add_path = bool(self.path_var.get())
        self.auth_token = secrets.token_urlsafe(32) if self.install_worker else None
        
        threading.Thread(target=self.install_process, daemon=True).start()
        
    def update_progress(self, val, text):
        self.progress_updates.put((val, text, False))

    def _poll_progress_updates(self):
        try:
            while True:
                val, text, finished = self.progress_updates.get_nowait()
                self.progress.set(val)
                self.status_lbl.configure(text=text)
                if finished:
                    self.after(1000, self.finish_and_self_destruct)
        except queue.Empty:
            pass
        self.after(100, self._poll_progress_updates)
        
    def install_process(self):
        try:
            self.update_progress(0.05, "Cerrando procesos antiguos...")
            subprocess.run(["taskkill", "/f", "/im", "ghost_terminal.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "p2p_node.exe"], capture_output=True)
            time.sleep(1)

            self.update_progress(0.1, "Creando directorios seguros...")
            os.makedirs(self.install_dir, exist_ok=True)
            time.sleep(0.6)
            
            self.update_progress(0.3, "Extrayendo Ghost Terminal (esto tarda un poco)...")

            # Los tres ejecutables comparten un unico runtime dentro de
            # GhostTerminal/_internal, asi que se copia el arbol entero. Copiar
            # solo los .exe dejaria una instalacion que no arranca.
            bundle_src = self.find_resource("GhostTerminal")
            shutil.copytree(bundle_src, self.install_dir, dirs_exist_ok=True)

            # Con el runtime compartido, omitir el worker ya no ahorra espacio
            # (su .exe son unos pocos MB); la casilla decide si se registra en
            # el arranque, no si se copia.
            if not self.install_worker:
                worker_exe = os.path.join(self.install_dir, "p2p_node.exe")
                if os.path.exists(worker_exe):
                    os.remove(worker_exe)

            self.update_progress(0.6, "Verificando la instalación...")
            required = ["ghost_terminal.exe", "uninstaller.exe"]
            missing = [name for name in required
                       if not os.path.exists(os.path.join(self.install_dir, name))]
            if missing:
                raise FileNotFoundError(
                    f"La copia quedó incompleta, faltan: {', '.join(missing)}")
            time.sleep(0.5)

            self.update_progress(0.65, "Comprobando puertos disponibles...")
            self.resolve_ports()

            self.update_progress(0.7, "Registrando en Windows y Auto-Arranque...")
            self.create_shortcuts_and_registry()

            if self.install_worker:
                self.update_progress(0.75, "Activando donación de recursos al arrancar Windows...")
                self.register_worker_autostart()
            
            if self.add_path:
                self.update_progress(0.8, "Añadiendo a las variables de entorno PATH...")
                self.add_to_path()
                
            time.sleep(0.6)
            self.progress_updates.put((1.0, "Instalación Completada. Iniciando...", True))
            
        except Exception as e:
            self.update_progress(0, f"Error crítico: {e}")
            import traceback
            traceback.print_exc()

    def resolve_ports(self):
        """Elige puertos libres y los deja escritos en la configuración.

        El worker y el cliente leen ese fichero al arrancar, así que el usuario
        no tiene que editar la línea de auto-arranque a mano si el 8001 o el
        8000 ya estaban ocupados por otra aplicación.
        """
        requested_worker = self.settings["worker_port"]
        requested_client = self.settings["client_port"]
        self.worker_port = find_free_port(requested_worker)
        client_port = find_free_port(requested_client)
        if self.worker_port == client_port:  # no pueden coincidir
            self.worker_port = find_free_port(client_port + 1)

        if self.worker_port != requested_worker:
            self.update_progress(0.66, f"Puerto {requested_worker} ocupado; se usará el {self.worker_port}.")
            time.sleep(0.8)

        try:
            self.settings.update({"worker_port": self.worker_port, "client_port": client_port})
            save_config(self.settings)
        except OSError as exc:
            # No es fatal: el nodo tiene su propio fallback de puerto al arrancar.
            print(f"[WARN] No se pudo guardar la configuración: {exc}")

    def add_to_path(self):
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            try:
                current_path, _ = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_path = ""
            entries = [entry for entry in current_path.split(os.pathsep) if entry]
            if not any(os.path.normcase(entry) == os.path.normcase(self.install_dir) for entry in entries):
                entries.append(self.install_dir)
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, os.pathsep.join(entries))

    def create_shortcuts_and_registry(self):
        target_path = os.path.join(self.install_dir, "ghost_terminal.exe")
        uninstaller_path = os.path.join(self.install_dir, "uninstaller.exe")
        shortcut_script = (
            "$ws=New-Object -ComObject WScript.Shell;"
            "$desktop=[Environment]::GetFolderPath('Desktop');"
            "$link=$ws.CreateShortcut((Join-Path $desktop 'Ghost Terminal.lnk'));"
            "$link.TargetPath=$env:GHOST_TARGET;$link.Save();"
            "$unlink=$ws.CreateShortcut((Join-Path $desktop 'Uninstall Ghost Terminal.lnk'));"
            "$unlink.TargetPath=$env:GHOST_UNINSTALLER;$unlink.Save()"
        )
        env = os.environ.copy()
        env.update({"GHOST_TARGET": target_path, "GHOST_UNINSTALLER": uninstaller_path})
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", shortcut_script],
            env=env,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
        ) as key:
            winreg.SetValueEx(key, "GhostTerminal", 0, winreg.REG_SZ, f'"{target_path}"')

    @staticmethod
    def _broadcast_environment_change():
        """Escribir en HKCU\\Environment no basta: sin WM_SETTINGCHANGE la
        variable no existe para los procesos ya arrancados, y el worker moriría
        con "--auth-token es obligatorio" sin dejar rastro visible."""
        try:
            import ctypes
            HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001A, 0x0002
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0,
                ctypes.c_wchar_p("Environment"), SMTO_ABORTIFHUNG, 3000, None,
            )
        except Exception:
            pass

    def register_worker_autostart(self):
        """Registra p2p_node.exe en el arranque de Windows para que el nodo done
        recursos automáticamente. Sin esta entrada, marcar la casilla "Instalar
        AI Torrent Worker" solo copiaba el ejecutable sin arrancarlo nunca."""
        worker_path = os.path.join(self.install_dir, "p2p_node.exe")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            winreg.SetValueEx(key, "AI_TORRENT_AUTH_TOKEN", 0, winreg.REG_SZ, self.auth_token)
        self._broadcast_environment_change()
        # --insecure-no-tls es obligatorio y explícito: una instalación normal no
        # dispone de un certificado emitido por la CA del enjambre, así que el
        # worker escucha en claro y el token viaja sin cifrar. Para desplegar con
        # TLS hay que emitir el certificado (src/gen_certs.py) y añadir
        # --tls-cert/--tls-key/--tls-ca a esta línea.
        command = (f'"{worker_path}" --host 0.0.0.0 --port {self.worker_port} --layers 8-15 '
                   f'--enable-tracker --insecure-no-tls')
        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
        ) as key:
            winreg.SetValueEx(key, "AITorrentWorker", 0, winreg.REG_SZ, command)

    # ------------------------------------------------ redimensionado

    def build_resize_grips(self):
        """Bordes activos.

        Una ventana con `overrideredirect(True)` pierde el marco del sistema y
        con él la posibilidad de redimensionar, así que hay que reimplementarla:
        ocho tiras finas y transparentes en los bordes y esquinas.
        """
        specs = [
            ("n",  dict(relx=0, rely=0, relwidth=1, height=GRIP), "sb_v_double_arrow"),
            ("s",  dict(relx=0, rely=1, y=-GRIP, relwidth=1, height=GRIP), "sb_v_double_arrow"),
            ("w",  dict(relx=0, rely=0, width=GRIP, relheight=1), "sb_h_double_arrow"),
            ("e",  dict(relx=1, x=-GRIP, rely=0, width=GRIP, relheight=1), "sb_h_double_arrow"),
            ("nw", dict(relx=0, rely=0, width=GRIP * 2, height=GRIP * 2), "size_nw_se"),
            ("ne", dict(relx=1, x=-GRIP * 2, rely=0, width=GRIP * 2, height=GRIP * 2), "size_ne_sw"),
            ("sw", dict(relx=0, rely=1, y=-GRIP * 2, width=GRIP * 2, height=GRIP * 2), "size_ne_sw"),
            ("se", dict(relx=1, x=-GRIP * 2, rely=1, y=-GRIP * 2, width=GRIP * 2, height=GRIP * 2), "size_nw_se"),
        ]
        self.grips = []
        for edge, placement, cursor in specs:
            grip = ctk.CTkFrame(self, width=10, height=10, fg_color="#00FF41", corner_radius=0)
            grip.place(**placement)
            try:
                grip.configure(cursor=cursor)
            except Exception:
                pass
            grip.bind("<Button-1>", lambda event, e=edge: self.start_resize(event, e))
            grip.bind("<B1-Motion>", self.do_resize)
            grip.bind("<ButtonRelease-1>", lambda event: setattr(self, "_resize_state", None))
            # Casi invisibles hasta que el ratón pasa por encima.
            grip.bind("<Enter>", lambda event, g=grip: g.configure(fg_color="#00FF41"))
            grip.bind("<Leave>", lambda event, g=grip: g.configure(fg_color="#0A0A0A"))
            grip.configure(fg_color="#0A0A0A")
            self.grips.append(grip)

    def start_resize(self, event, edge):
        self._resize_state = {
            "edge": edge,
            "x_root": event.x_root,
            "y_root": event.y_root,
            "width": self.winfo_width(),
            "height": self.winfo_height(),
            "x": self.winfo_rootx(),
            "y": self.winfo_rooty(),
        }

    def do_resize(self, event):
        state = self._resize_state
        if not state:
            return
        dx = event.x_root - state["x_root"]
        dy = event.y_root - state["y_root"]
        width, height = state["width"], state["height"]
        x, y = state["x"], state["y"]
        edge = state["edge"]

        if "e" in edge:
            width = state["width"] + dx
        if "w" in edge:
            width = state["width"] - dx
            x = state["x"] + dx
        if "s" in edge:
            height = state["height"] + dy
        if "n" in edge:
            height = state["height"] - dy
            y = state["y"] + dy

        # Al topar con el mínimo, el borde opuesto no debe seguir desplazándose.
        if width < MIN_WIDTH:
            if "w" in edge:
                x -= MIN_WIDTH - width
            width = MIN_WIDTH
        if height < MIN_HEIGHT:
            if "n" in edge:
                y -= MIN_HEIGHT - height
            height = MIN_HEIGHT
        width = min(width, self.winfo_screenwidth())
        height = min(height, self.winfo_screenheight())

        self.geometry(f"{int(width)}x{int(height)}+{int(x)}+{int(y)}")

    def _on_configure(self, event):
        """Reescala el fondo cuando cambia el tamaño, con anti-rebote.

        Sin esto la imagen conserva su tamaño original y aparecen bandas negras
        (o se recorta) al agrandar la ventana.
        """
        if getattr(event, "widget", None) is not self:
            return
        if self._bg_redraw_job is not None:
            self.after_cancel(self._bg_redraw_job)
        self._bg_redraw_job = self.after(BG_REDRAW_MS, self._rescale_background)

    def _rescale_background(self):
        self._bg_redraw_job = None
        image = getattr(self, "bg_image", None)
        if image is None:
            return
        try:
            image.configure(size=(max(1, self.winfo_width()), max(1, self.winfo_height())))
        except Exception:
            pass

    def get_pos(self, event):
        self._x = event.x
        self._y = event.y

    def move_window(self, event):
        self.geometry(f"+{event.x_root - self._x}+{event.y_root - self._y}")

    def finish_and_self_destruct(self):
        if not getattr(sys, 'frozen', False):
            sys.exit(0)
            
        my_exe = sys.executable
        target_exe = os.path.join(self.install_dir, "ghost_terminal.exe")
        
        cleanup_script = (
            "Start-Sleep -Seconds 2;"
            "Remove-Item -LiteralPath $env:GHOST_SETUP -Force -ErrorAction SilentlyContinue;"
            "Start-Process -FilePath $env:GHOST_TARGET"
        )
        env = os.environ.copy()
        env.update({"GHOST_SETUP": my_exe, "GHOST_TARGET": target_exe})
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cleanup_script],
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        sys.exit(0)

if __name__ == "__main__":
    try:
        app = SetupWizard()
        app.mainloop()
    except Exception as e:
        import traceback
        print("CRITICAL ERROR:")
        traceback.print_exc()
        input("\nHa ocurrido un error fatal. Presiona Enter para salir...")
