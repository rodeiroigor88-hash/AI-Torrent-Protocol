r"""Firma Authenticode de los binarios de Ghost Terminal.

Smart App Control (Windows 11) y SmartScreen bloquean cualquier .exe sin firmar
o "no conocido". La unica solucion real para distribuir es firmar con un
certificado reconocido por Microsoft (una firma autofirmada NO vale para SAC).
Ver docs/firma-codigo.md para elegir certificado.

La firma es OPT-IN por variables de entorno: si no hay credencial configurada,
`build.py` sigue generando los .exe sin firmar (con un aviso), para no romper el
build de quien todavia no tiene certificado. Formas de credencial admitidas:

  # 1) Fichero PFX (certificado OV/EV exportado)
  set GHOST_SIGN_PFX=C:\ruta\cert.pfx
  set GHOST_SIGN_PFX_PASSWORD=...             (opcional)

  # 2) Certificado ya instalado en el almacen de Windows (por huella SHA-1)
  set GHOST_SIGN_THUMBPRINT=AB12...CD

  # 3) Azure Trusted Signing (Microsoft.Trusted.Signing.Client)
  set GHOST_SIGN_AZURE_DLIB=C:\...\Azure.CodeSigning.Dlib.dll
  set GHOST_SIGN_AZURE_METADATA=C:\...\metadata.json

Uso directo (sin recompilar), util para re-firmar:
  python sign.py dist/ghost_terminal.exe
  python sign.py --verify "dist/Ghost Terminal Setup/Ghost Terminal Setup.exe"
"""

import glob
import os
import shutil
import subprocess
import sys

# Servidor de sellado de tiempo RFC-3161: sin el, la firma deja de ser valida
# cuando caduca el certificado. Se puede cambiar con GHOST_SIGN_TIMESTAMP.
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"


class SigningError(RuntimeError):
    pass


def find_signtool() -> str:
    """Localiza signtool.exe (viene con el Windows SDK)."""
    found = shutil.which("signtool")
    if found:
        return found
    patterns = [
        r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\*\bin\*\x64\signtool.exe",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if candidates:
        # La ruta mas alta suele ser la version mas reciente del SDK.
        return sorted(candidates)[-1]
    raise SigningError(
        "No se encontro signtool.exe. Instala el Windows SDK "
        "(https://developer.microsoft.com/windows/downloads/windows-sdk/) "
        "o anade signtool al PATH."
    )


def credential_args():
    """Argumentos de credencial para signtool, o None si no hay nada configurado."""
    pfx = os.environ.get("GHOST_SIGN_PFX")
    thumbprint = os.environ.get("GHOST_SIGN_THUMBPRINT")
    azure_dlib = os.environ.get("GHOST_SIGN_AZURE_DLIB")
    azure_meta = os.environ.get("GHOST_SIGN_AZURE_METADATA")

    if pfx:
        if not os.path.isfile(pfx):
            raise SigningError(f"GHOST_SIGN_PFX apunta a un fichero inexistente: {pfx}")
        args = ["/f", pfx]
        password = os.environ.get("GHOST_SIGN_PFX_PASSWORD")
        if password:
            args += ["/p", password]
        return args
    if thumbprint:
        return ["/sha1", thumbprint.replace(" ", "")]
    if azure_dlib and azure_meta:
        return ["/dlib", azure_dlib, "/dmdf", azure_meta]
    return None


def is_configured() -> bool:
    try:
        return credential_args() is not None
    except SigningError:
        return True  # configurado pero mal: que el build lo reporte al firmar


def sign_file(path: str, signtool: str = None, credential=None) -> None:
    if not os.path.exists(path):
        raise SigningError(f"No existe el fichero a firmar: {path}")
    signtool = signtool or find_signtool()
    credential = credential if credential is not None else credential_args()
    if credential is None:
        raise SigningError("No hay credencial de firma configurada (ver cabecera de sign.py).")

    timestamp = os.environ.get("GHOST_SIGN_TIMESTAMP", DEFAULT_TIMESTAMP_URL)
    command = [signtool, "sign", "/fd", "sha256", "/tr", timestamp, "/td", "sha256"]
    command += credential + [path]

    # No imprimimos el comando: podria llevar la contrasena del PFX en claro.
    printable = os.path.basename(path)
    print(f"    [firma] {printable} ...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SigningError(
            f"signtool fallo al firmar {printable} (codigo {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def verify_file(path: str, signtool: str = None) -> bool:
    signtool = signtool or find_signtool()
    # /pa = usar la politica de Authenticode por defecto.
    result = subprocess.run([signtool, "verify", "/pa", "/v", path],
                            capture_output=True, text=True)
    ok = result.returncode == 0
    print(f"    [verificacion] {os.path.basename(path)}: "
          f"{'FIRMADO Y VALIDO' if ok else 'SIN FIRMA VALIDA'}")
    if not ok:
        print("        " + (result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""))
    return ok


def sign_many(paths, required: bool = False) -> bool:
    """Firma una lista de ficheros. Devuelve True si se firmo todo.

    Si no hay credencial configurada: aviso y se devuelve False (build sin firma),
    salvo `required=True`, que aborta.
    """
    if not is_configured():
        message = (
            "\n[!] FIRMA DE CODIGO NO CONFIGURADA.\n"
            "    Los .exe quedaran SIN FIRMAR y Smart App Control / SmartScreen\n"
            "    los bloqueara en otras maquinas. Configura un certificado\n"
            "    (ver docs/firma-codigo.md) o define GHOST_SIGN_PFX / _THUMBPRINT.\n"
        )
        if required:
            raise SigningError(message)
        print(message)
        return False

    signtool = find_signtool()
    credential = credential_args()
    for path in paths:
        sign_file(path, signtool=signtool, credential=credential)
    for path in paths:
        verify_file(path, signtool=signtool)
    return True


def _cli():
    args = [a for a in sys.argv[1:] if a != "--verify"]
    if not args:
        print(__doc__)
        return 1
    if "--verify" in sys.argv:
        signtool = find_signtool()
        all_ok = all(verify_file(path, signtool=signtool) for path in args)
        return 0 if all_ok else 1
    try:
        sign_many(args, required=True)
    except SigningError as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
