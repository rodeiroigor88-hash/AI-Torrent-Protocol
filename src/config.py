"""Configuracion persistente compartida por el cliente, el worker y el instalador.

Vive en `%LOCALAPPDATA%\\GhostTerminal\\config.json` (o `~/.tokentorrent/` fuera de
Windows), junto al log del worker. Existe para que el usuario pueda cambiar el
atajo de teclado, la URL del tracker y cuantos recursos dona **sin tocar el
codigo ni recompilar**.

Todo valor que entra se valida y se recorta a su rango: el fichero lo edita un
humano y lo lee un proceso que corre sin consola, asi que un valor absurdo debe
degradar al de por defecto, nunca reventar el arranque.
"""

import json
import logging
import os
import tempfile
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# A production tracker must be supplied explicitly. The old public URL was a
# redirecting placeholder and caused nodes to silently enroll against a 404.
# Keep local development deterministic and require deployment configuration for
# any real swarm.
DEFAULT_TRACKER_URL = "https://api.tokentorrent.es/ai-torrent"

# 15 s de latido para que el tracker pueda expulsar a un nodo fantasma a los
# 45 s (tres latidos perdidos) sin desconectar a nodos sanos por un fallo suelto.
HEARTBEAT_INTERVAL = 15

DEFAULTS = {
    # Interfaz
    "hotkey": "f12",
    "extra_hotkeys": ["ctrl+~", "ctrl+º", "ctrl+space"],
    # Red
    "tracker_url": DEFAULT_TRACKER_URL,
    "client_port": 8000,
    "worker_port": 8001,
    # Donacion de recursos (QoS)
    "max_ram_percent": 50,      # % de la RAM total que el worker puede ocupar
    "cpu_cores": 0,             # 0 = automatico (la mitad de los nucleos)
    "cpu_pause_threshold": 80,  # % de CPU global a partir del cual se pausa
}

_INT_RANGES = {
    "client_port": (1, 65535),
    "worker_port": (1, 65535),
    "max_ram_percent": (5, 95),
    "cpu_cores": (0, 256),
    "cpu_pause_threshold": (10, 100),
}


def config_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "GhostTerminal")
    return os.path.join(os.path.expanduser("~"), ".tokentorrent")


def config_path() -> str:
    override = os.environ.get("TOKENTORRENT_CONFIG")
    if override:
        return override
    return os.path.join(config_dir(), "config.json")


def _clean_hotkey(value, fallback: str) -> str:
    """Un atajo es una cadena corta tipo `ctrl+shift+g`; nada mas."""
    if not isinstance(value, str):
        return fallback
    value = value.strip().lower()
    if not value or len(value) > 64:
        return fallback
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789+ _-<>[]{}ºçñ~`,.;'\"/\\|!@#$%^&*()")
    if any(char not in allowed for char in value):
        return fallback
    return value


def _clean_tracker_url(value, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    value = value.strip().rstrip("/")
    if not value or len(value) > 512:
        return fallback
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return fallback
    if parts.username or parts.password:
        return fallback
    return value


def sanitize(raw: dict) -> dict:
    """Devuelve una configuracion completa y valida a partir de datos crudos."""
    config = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return config

    config["hotkey"] = _clean_hotkey(raw.get("hotkey"), DEFAULTS["hotkey"])
    config["tracker_url"] = _clean_tracker_url(raw.get("tracker_url"), DEFAULTS["tracker_url"])

    extra = raw.get("extra_hotkeys")
    if isinstance(extra, list):
        cleaned = []
        for item in extra[:8]:
            hotkey = _clean_hotkey(item, "")
            if hotkey and hotkey not in cleaned:
                cleaned.append(hotkey)
        config["extra_hotkeys"] = cleaned

    for key, (low, high) in _INT_RANGES.items():
        value = raw.get(key, DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = DEFAULTS[key]
        config[key] = max(low, min(high, value))

    return config


def load_config() -> dict:
    path = config_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return dict(DEFAULTS)
    except (OSError, ValueError) as exc:
        logger.warning("Config ilegible en %s (%s); se usan los valores por defecto.", path, exc)
        return dict(DEFAULTS)
    return sanitize(raw)


def save_config(config: dict) -> str:
    """Escritura atomica: un corte de luz a media escritura no debe dejar un
    JSON truncado que impida arrancar la proxima vez."""
    path = config_path()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    clean = sanitize(config)

    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=".config-", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(clean, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


def update_config(**changes) -> dict:
    config = load_config()
    config.update(changes)
    save_config(config)
    return config


def tracker_url() -> str:
    """URL del tracker: variable de entorno > config > valor por defecto."""
    from_env = os.environ.get("TOKENTORRENT_TRACKER_URL")
    if from_env:
        return _clean_tracker_url(from_env, DEFAULT_TRACKER_URL)
    return load_config()["tracker_url"]


def tracker_token() -> str:
    """Token de enrollment del tracker, solo desde variable de entorno."""
    return os.environ.get("TOKENTORRENT_NODE_SECRET", "").strip()
