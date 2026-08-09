import customtkinter as ctk
import keyboard
import logging
import threading
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chat_agent import AgenticChat
from src.config import DEFAULTS, load_config, save_config

logger = logging.getLogger(__name__)

# Configuraciones de estética Cypherpunk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

GHOST_GREEN = "#00FF41"


def build_ghost_icon(size: int = 64):
    """Dibuja el fantasmita verde de la bandeja del sistema.

    Se dibuja en vez de cargar `ghost.ico` para no depender de que PyInstaller
    empaquete el recurso: si el icono falta, el usuario pierde el único indicio
    visible de que la aplicación está corriendo.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    green = (0, 255, 65, 255)
    dark = (5, 5, 5, 255)

    margin = size // 8
    body_bottom = size - margin * 2
    # Cabeza redonda + cuerpo recto
    draw.ellipse([margin, margin, size - margin, size - margin * 3], fill=green)
    draw.rectangle([margin, size // 2 - margin, size - margin, body_bottom], fill=green)
    # Faldón ondulado (tres picos)
    wave = (size - margin * 2) / 3.0
    for index in range(3):
        left = margin + wave * index
        draw.polygon(
            [(left, body_bottom), (left + wave / 2, body_bottom + margin), (left + wave, body_bottom)],
            fill=green,
        )
    # Ojos
    eye_r = max(2, size // 12)
    eye_y = size // 2 - margin
    draw.ellipse([size // 2 - eye_r * 3, eye_y - eye_r, size // 2 - eye_r, eye_y + eye_r], fill=dark)
    draw.ellipse([size // 2 + eye_r, eye_y - eye_r, size // 2 + eye_r * 3, eye_y + eye_r], fill=dark)
    return image


class SettingsWindow(ctk.CTkToplevel):
    """Panel de ajustes: atajo de teclado y URL del tracker.

    Sin esto había que recompilar para apuntar a un tracker local, y el usuario
    no tenía forma de cambiar un atajo que le chocara con otra aplicación.
    """

    def __init__(self, master, on_saved):
        super().__init__(master)
        self.on_saved = on_saved
        self.config_data = load_config()

        self.title("Ghost Terminal // Ajustes")
        self.geometry("520x340")
        self.configure(fg_color="#0A0A0A")
        self.attributes("-topmost", True)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        ctk.CTkLabel(self, text="AJUSTES", font=("Consolas", 20, "bold"),
                     text_color=GHOST_GREEN).pack(pady=(20, 4))
        ctk.CTkLabel(self, text="Los cambios se aplican al instante, sin reiniciar.",
                     font=("Consolas", 11), text_color="#777777").pack(pady=(0, 12))

        self.hotkey_entry = self._field(
            "Atajo principal", self.config_data["hotkey"],
            "Ejemplos: f12, ctrl+shift+g, alt+space",
        )
        self.extra_entry = self._field(
            "Atajos alternativos", ", ".join(self.config_data["extra_hotkeys"]),
            "Separados por comas. Déjalo vacío si no quieres más.",
        )
        self.tracker_entry = self._field(
            "URL del Tracker", self.config_data["tracker_url"],
            "Para pruebas locales: http://127.0.0.1:5000/ai-torrent",
        )

        self.status_lbl = ctk.CTkLabel(self, text="", font=("Consolas", 11), text_color="#FF9900")
        self.status_lbl.pack(pady=(6, 0))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(pady=12)
        ctk.CTkButton(buttons, text="Cancelar", width=110, fg_color="#333333",
                      hover_color="#555555", command=self.destroy).pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Guardar", width=110, fg_color="#005511",
                      text_color=GHOST_GREEN, hover_color="#00CC33",
                      command=self.save).pack(side="left", padx=8)

        self.after(120, self.focus_force)

    def _field(self, label, value, hint):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=30, pady=4)
        ctk.CTkLabel(frame, text=label, font=("Consolas", 12), text_color="#CCCCCC",
                     width=150, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(frame, font=("Consolas", 12), fg_color="#111111",
                             text_color="#FFFFFF", width=300)
        entry.pack(side="left")
        entry.insert(0, value)
        ctk.CTkLabel(self, text=hint, font=("Consolas", 10), text_color="#666666",
                     anchor="w").pack(fill="x", padx=(190, 30))
        return entry

    def save(self):
        extras = [item.strip() for item in self.extra_entry.get().split(",") if item.strip()]
        requested = {
            "hotkey": self.hotkey_entry.get().strip(),
            "extra_hotkeys": extras,
            "tracker_url": self.tracker_entry.get().strip(),
        }
        merged = dict(self.config_data)
        merged.update(requested)
        try:
            save_config(merged)
        except OSError as exc:
            self.status_lbl.configure(text=f"No se pudo guardar: {exc}", text_color="#FF4444")
            return

        # `save_config` sanea: si el usuario escribió algo inválido se lo decimos
        # en vez de guardarlo en silencio y dejarle con un atajo que no funciona.
        stored = load_config()
        rejected = [name for name, value in requested.items()
                    if name != "extra_hotkeys" and stored[name] != value]
        self.on_saved(stored)
        if rejected:
            self.status_lbl.configure(
                text=f"Valor no válido en: {', '.join(rejected)}. Se mantuvo el anterior.",
                text_color="#FF9900")
            for name, entry in (("hotkey", self.hotkey_entry), ("tracker_url", self.tracker_entry)):
                entry.delete(0, "end")
                entry.insert(0, stored[name])
            return
        self.status_lbl.configure(text="Guardado.", text_color=GHOST_GREEN)
        self.after(700, self.destroy)


class GhostTerminal(ctk.CTk):
    def __init__(self, chat_agent: AgenticChat, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.chat_agent = chat_agent
        self.loop = loop
        self.settings = load_config()
        self._hotkey_handles = []
        self._settings_window = None
        self.tray_icon = None

        self.title("Ghost Terminal")

        # Sin bordes y translúcido
        self.overrideredirect(True)
        self.attributes("-alpha", 0.90)
        self.configure(fg_color="#0A0A0A") # Negro profundo
        self.attributes("-topmost", True)

        # Posicionamiento: Panel superior deslizante
        screen_width = self.winfo_screenwidth()
        width = int(screen_width * 0.6)
        height = 500
        x = (screen_width // 2) - (width // 2)
        self.geometry(f"{width}x{height}+{x}+0")

        self.is_visible = False

        # Layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top Bar para el menú
        self.top_bar = ctk.CTkFrame(self, height=30, fg_color="transparent")
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Tres puntitos (menú)
        self.menu_btn = ctk.CTkButton(self.top_bar, text="⋮", font=("Consolas", 18, "bold"), width=30, height=30,
                                      fg_color="transparent", text_color="#AAAAAA", hover_color="#222222",
                                      command=self.show_menu)
        self.menu_btn.pack(side="right")

        # TextBox (Terminal Output)
        self.output_text = ctk.CTkTextbox(
            self,
            fg_color="transparent",
            text_color=GHOST_GREEN, # Hacker Green
            font=("Consolas", 16),
            wrap="word"
        )
        self.output_text.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.output_text.insert(
            "0.0",
            "SYSTEM_BOOT: THE GHOST TERMINAL V1.0\n"
            "STATUS: CONNECTED TO AI TORRENT SWARM\n"
            f"PRESS [{self.settings['hotkey'].upper()}] TO HIDE/SHOW\n\n"
        )
        self.output_text.configure(state="disabled")

        # InputBox (Terminal Input)
        self.input_entry = ctk.CTkEntry(
            self,
            placeholder_text="> type your prompt...",
            font=("Consolas", 16),
            fg_color="#1A1A1A",
            text_color="#FFFFFF",
            border_width=1,
            border_color="#333333"
        )
        self.input_entry.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_entry.bind("<Return>", self.on_enter)
        self.input_entry.configure(state="disabled")
        self.after(100, self._check_backend_status)

        # Bindings para cerrar o esconder
        self.bind("<FocusOut>", self.on_focus_out)
        for widget in (self, self.input_entry, self.output_text):
            widget.bind("<Escape>", lambda e: self.trigger_toggle())
            widget.bind("<F12>", lambda e: self.trigger_toggle())

        self.register_hotkeys()
        self.start_tray_icon()

        self.withdraw() # Iniciar oculto

    # ------------------------------------------------------------ atajos

    def register_hotkeys(self):
        """(Re)registra los atajos globales según la configuración."""
        for handle in self._hotkey_handles:
            try:
                keyboard.remove_hotkey(handle)
            except (KeyError, ValueError):
                pass
        self._hotkey_handles = []

        combos = [self.settings["hotkey"]] + list(self.settings["extra_hotkeys"])
        registered = []
        for combo in combos:
            if combo in registered:
                continue
            try:
                self._hotkey_handles.append(keyboard.add_hotkey(combo, self.trigger_toggle))
                registered.append(combo)
            except Exception as exc:
                # Un atajo inválido no puede tumbar la aplicación: se avisa y se sigue.
                logger.warning("No se pudo registrar el atajo %r: %s", combo, exc)
                self.after(0, self.append_text, f"\n[AVISO] Atajo no válido: {combo}\n")

        if not self._hotkey_handles:
            fallback = DEFAULTS["hotkey"]
            try:
                self._hotkey_handles.append(keyboard.add_hotkey(fallback, self.trigger_toggle))
                self.after(0, self.append_text,
                           f"\n[AVISO] Ningún atajo válido; se restauró [{fallback.upper()}].\n")
            except Exception as exc:
                logger.error("Sin atajos globales disponibles: %s", exc)

    def apply_settings(self, settings: dict):
        """Aplica en caliente lo que se guardó en el panel de Ajustes."""
        self.settings = settings
        self.register_hotkeys()
        self.chat_agent.tracker_url = settings["tracker_url"]
        # Una ruta cacheada del tracker viejo no vale para el nuevo.
        self.chat_agent.route = None
        self.chat_agent.primary_node_url = None
        self.append_text(f"\n[CONFIG] Atajo: {settings['hotkey'].upper()} | "
                         f"Tracker: {settings['tracker_url']}\n")

    # ------------------------------------------------------- bandeja

    def start_tray_icon(self):
        """Icono junto al reloj: sin él, quien olvide el atajo cree que la app
        no arrancó y no tiene forma de recuperarla ni de cerrarla."""
        try:
            import pystray
        except Exception as exc:
            logger.warning("Bandeja del sistema no disponible: %s", exc)
            return

        menu = pystray.Menu(
            pystray.MenuItem("Mostrar Terminal", self._tray_show, default=True),
            pystray.MenuItem("Ajustes", self._tray_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Salir", self._tray_quit),
        )
        try:
            self.tray_icon = pystray.Icon("ghost_terminal", build_ghost_icon(),
                                          "Ghost Terminal", menu)
            threading.Thread(target=self.tray_icon.run, daemon=True).start()
        except Exception as exc:
            logger.warning("No se pudo crear el icono de bandeja: %s", exc)
            self.tray_icon = None

    # pystray llama a estos callbacks desde SU hilo: Tkinter no es thread-safe,
    # así que todo se delega al hilo de la interfaz con `after`.
    def _tray_show(self, icon=None, item=None):
        self.after(0, self.show_window)

    def _tray_settings(self, icon=None, item=None):
        self.after(0, self.open_settings)

    def _tray_quit(self, icon=None, item=None):
        self.after(0, self.quit_app)

    def show_window(self):
        if not self.is_visible:
            self.toggle_visibility()
        else:
            self.lift()
            self.focus_force()

    def quit_app(self):
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        if self.loop.is_running():
            # Se le da un margen corto al cierre limpio: si no, quedan sockets
            # abiertos y el puerto del callback sigue ocupado un rato.
            try:
                future = asyncio.run_coroutine_threadsafe(self.chat_agent.teardown(), self.loop)
                future.result(timeout=3)
            except Exception as exc:
                logger.warning("Cierre del backend incompleto: %s", exc)
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()
        # El hilo de pystray no es demonio en todas las plataformas: os._exit
        # garantiza que no quede el proceso vivo sin ventana ni icono.
        os._exit(0)

    # ---------------------------------------------------------- menús

    def open_settings(self):
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.focus_force()
            return
        self.show_window()
        self._settings_window = SettingsWindow(self, self.apply_settings)

    def show_menu(self):
        # Crear un pequeño menú popup
        menu = ctk.CTkToplevel(self)
        menu.title("")
        menu.overrideredirect(True)
        menu.geometry(f"140x120+{self.winfo_rootx() + self.winfo_width() - 150}+{self.winfo_rooty() + 40}")
        menu.configure(fg_color="#1A1A1A")
        menu.attributes("-topmost", True)

        def close_app():
            menu.destroy()
            self.quit_app()

        def hide_app():
            menu.destroy()
            self.trigger_toggle()

        def open_settings():
            menu.destroy()
            self.open_settings()

        btn_settings = ctk.CTkButton(menu, text="Ajustes", fg_color="transparent",
                                     hover_color="#333333", command=open_settings)
        btn_settings.pack(fill="x", pady=(5, 0), padx=5)

        btn_hide = ctk.CTkButton(menu, text=f"Ocultar ({self.settings['hotkey'].upper()})",
                                 fg_color="transparent", hover_color="#333333", command=hide_app)
        btn_hide.pack(fill="x", pady=(5, 0), padx=5)

        btn_close = ctk.CTkButton(menu, text="Salir del todo", fg_color="transparent",
                                  text_color="#FF4444", hover_color="#330000", command=close_app)
        btn_close.pack(fill="x", pady=5, padx=5)

        # Cerrar el menú si se hace click fuera
        menu.bind("<FocusOut>", lambda e: menu.destroy())
        menu.focus_force()

    def on_focus_out(self, event):
        def _check_focus():
            # Con el panel de Ajustes abierto el foco sale de la ventana
            # principal; ocultarla dejaría los Ajustes huérfanos.
            if self._settings_window is not None and self._settings_window.winfo_exists():
                return
            fw = self.focus_get()
            # Si el foco ya no pertenece a nuestra app (es None), ocultamos
            if fw is None and self.is_visible:
                self.trigger_toggle()
        self.after(100, _check_focus)

    def _check_backend_status(self):
        if getattr(self.chat_agent, "setup_error", None):
            self.append_text(f"\n[ERROR DE INICIALIZACION] {self.chat_agent.setup_error}\n")
            return
        if self.chat_agent.client_model is not None and self.chat_agent.session is not None:
            self.input_entry.configure(state="normal")
            return
        self.after(100, self._check_backend_status)

    def trigger_toggle(self):
        # Debounce para evitar que el hotkey global y el bind local salten a la vez
        now = time.time()
        if hasattr(self, '_last_toggle') and now - self._last_toggle < 0.2:
            return
        self._last_toggle = now

        # Tkinter no es thread-safe, delegamos la acción al hilo principal de la UI
        self.after(0, self.toggle_visibility)

    def toggle_visibility(self):
        if self.is_visible:
            self.withdraw()
            self.is_visible = False
        else:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.is_visible = True

            # Un pequeño delay asegura que Windows le dé el foco real al input
            def _force_focus():
                self.focus_force()
                self.input_entry.focus()
                self.input_entry.focus_set()

            self.after(50, _force_focus)

    def append_text(self, text: str, color: str = None):
        self.output_text.configure(state="normal")
        # CustomTkinter no soporta colores por tags fácilmente en CTkTextbox base de forma nativa sin tkinter puro,
        # así que nos conformamos con el verde hacker.
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def on_enter(self, event):
        user_input = self.input_entry.get().strip()
        if not user_input: return

        # Comandos locales
        if user_input.lower() in ("exit", "quit", "salir"):
            self.quit_app()
            return

        self.input_entry.delete(0, "end")
        self.input_entry.configure(state="disabled")
        self.append_text(f"\n> {user_input}\n")

        def _yield_token(token):
            self.after(0, self.append_text, token)

        async def _run_generation():
            try:
                await self.chat_agent.generate_response(user_input, _yield_token)
            except Exception as exc:
                self.after(0, self.append_text, f"\n[ERROR] {exc}\n")
            finally:
                self.after(0, self.append_text, "\n")
                self.after(0, lambda: self.input_entry.configure(state="normal"))

        asyncio.run_coroutine_threadsafe(_run_generation(), self.loop)

def start_asyncio_thread(chat_agent, loop):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(chat_agent.setup())
        loop.run_forever()
    except Exception as exc:
        chat_agent.setup_error = str(exc)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s] - %(message)s')
    settings = load_config()
    # La identidad TLS la deja el instalador en el entorno; sin ella el cliente
    # sigue funcionando en claro contra un enjambre local.
    chat = AgenticChat(model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        client_port=settings["client_port"],
                        auth_token=os.environ.get("AI_TORRENT_AUTH_TOKEN"),
                        tls_cert=os.environ.get("AI_TORRENT_TLS_CERT"),
                        tls_key=os.environ.get("AI_TORRENT_TLS_KEY"),
                        tls_ca=os.environ.get("AI_TORRENT_TLS_CA"),
                        tracker_url=settings["tracker_url"])

    # Crear un loop dedicado para el backend
    bg_loop = asyncio.new_event_loop()

    # Arrancar el backend P2P en un hilo secundario
    async_thread = threading.Thread(target=start_asyncio_thread, args=(chat, bg_loop), daemon=True)
    async_thread.start()

    # Arrancar la UI en el hilo principal
    app = GhostTerminal(chat, bg_loop)

    # Hacer que la aplicación se oculte temporalmente al inicio para cargar bien
    app.after(100, app.withdraw)
    print(f"\n[GHOST TERMINAL] Corriendo en segundo plano. "
          f"Pulsa {settings['hotkey'].upper()} o usa el icono de la bandeja.\n")

    app.mainloop()
