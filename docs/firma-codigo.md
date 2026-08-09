# Firma de código: pasar Smart App Control y SmartScreen

## El problema

Windows 11 (build reciente) trae **Control inteligente de aplicaciones**
(*Smart App Control*, SAC). Cuando está en modo *enforcement* **bloquea todo
ejecutable que no esté firmado o que Microsoft no reconozca como "conocido
bueno"**, y —a diferencia del SmartScreen clásico— **no ofrece un botón de
"Ejecutar de todas formas"**. Por eso `Ghost Terminal Setup.exe` no arranca.

Comprobado en la máquina de desarrollo:

```
Estado firma: NotSigned
Smart App Control: ACTIVADO / enforcement (VerifiedAndReputablePolicyState = 1)
```

No es un virus (Defender no puso nada en cuarentena): es puramente falta de
firma reconocida.

> Una firma **autofirmada NO sirve** para SAC. SAC no valida solo la cadena
> Authenticode; consulta la reputación de Microsoft (*Intelligent Security
> Graph*). Un certificado que solo tú confías no la satisface.

## Opciones de certificado

Ordenadas por relación coste / rapidez en dejar de ser bloqueado:

| Opción | Coste | Requisitos | ¿Pasa SAC ya? |
|---|---|---|---|
| **SignPath (plan OSS)** | Gratis | Proyecto open source que cumpla sus condiciones | Sí (usa un cert con reputación) |
| **Azure Trusted Signing** | ~10 $/mes | Identidad verificable de **3+ años** (individuo) u organización verificada | Sí, es el servicio nativo de Microsoft |
| **Certificado EV** (Sectigo/DigiCert) | ~300–600 $/año | Validación de identidad + token hardware/HSM | **Sí, inmediato** (máxima reputación) |
| **Certificado OV** estándar | ~150–250 $/año | Validación de identidad | Parcial: Authenticado válido, pero la reputación de SmartScreen **se gana con el tiempo**; SAC puede seguir bloqueando al principio |
| Autofirmado | Gratis | — | **No** (inútil para SAC) |

**Opción elegida para este proyecto: SignPath Foundation** (plan gratuito para
proyectos de código abierto). Alternativas si esa vía no prosperase: Azure
Trusted Signing, o un certificado EV si se quiere cero fricción desde el primer
binario.

### Requisitos de la SignPath Foundation

Al solicitarlo hay que tener en cuenta dos cosas que se comprueban **durante la
revisión**, no después:

1. **La *Download URL* debe mencionar que el proyecto usa SignPath.** Ojo: la
   página de *Releases* de GitHub **no** renderiza el README, así que si se
   apunta ahí y no hay ninguna release publicada con esa mención en sus notas,
   el revisor no encontrará nada. La atribución vive en la sección
   "Descargas y firma de código" del README, que es lo que se ve en la raíz del
   repositorio.
2. **La *Privacy Policy URL*** debe llevar a algo que describa qué datos maneja
   el software. La sección "Privacidad" del README cubre ese punto.

Al publicar la primera release, conviene repetir la frase de atribución a
SignPath en las notas de la release.

Lo que yo **no** puedo hacer por ti: comprar el certificado, superar la
validación de identidad ni instalar la credencial. Eso lo haces tú con el
proveedor. Lo que ya está listo es todo lo demás.

## Cómo firmar una vez tengas el certificado

`build.py` firma automáticamente si detecta una credencial en el entorno. Todo
lo gestiona `sign.py`. Elige **una** de estas tres formas:

**1) Fichero PFX** (certificado OV/EV exportado a `.pfx`):
```bash
set GHOST_SIGN_PFX=C:\ruta\a\tu-cert.pfx
set GHOST_SIGN_PFX_PASSWORD=tu-contraseña
```

**2) Certificado ya instalado** en el almacén de Windows (por huella SHA-1):
```bash
set GHOST_SIGN_THUMBPRINT=AB12CD34...
```

**3) Azure Trusted Signing** (paquete `Microsoft.Trusted.Signing.Client`):
```bash
set GHOST_SIGN_AZURE_DLIB=C:\ruta\Azure.CodeSigning.Dlib.dll
set GHOST_SIGN_AZURE_METADATA=C:\ruta\metadata.json
```

Y luego, simplemente:
```bash
python build.py
```

El orden es el correcto por diseño: primero firma `ghost_terminal.exe`,
`p2p_node.exe` y `uninstaller.exe`, **después** empaqueta el instalador (que los
embebe ya firmados) y por último firma `Ghost Terminal Setup.exe`, que es el
`.exe` que el usuario ejecuta primero.

### Firmar sin recompilar

Para re-firmar un binario ya construido, o comprobar el estado de firma:
```bash
python sign.py dist/ghost_terminal.exe
```
```bash
python sign.py --verify "dist/Ghost Terminal Setup/Ghost Terminal Setup.exe"
```

## Detalles que importan

- **Sellado de tiempo**: `sign.py` usa siempre un servidor RFC-3161
  (`http://timestamp.digicert.com` por defecto, cambiable con
  `GHOST_SIGN_TIMESTAMP`). Sin él, las firmas caducan cuando caduca el
  certificado. No lo quites.
- **Reputación con certificados nuevos**: un cert OV recién emitido firma
  válidamente pero puede seguir chocando con SmartScreen/SAC hasta acumular
  reputación (número de instalaciones limpias). EV y Azure Trusted Signing
  evitan esa espera.
- **La contraseña del PFX** se pasa a `signtool` pero `sign.py` nunca la
  imprime en el log del build. Aun así, evita dejarla en el historial del shell:
  prefiere la forma por huella (opción 2) con el cert en el almacén de usuario.
- **`requirements.txt`**: firmar no añade dependencias de Python; `signtool.exe`
  viene con el Windows SDK (ya detectado en esta máquina:
  `10.0.26100.0`).

## Mientras tanto, para probar en local

Si solo quieres verificar que la app funciona en **esta** máquina sin esperar al
certificado, tienes dos vías (ninguna la hago yo por ti, son decisiones tuyas):

- **Ejecutar desde el código fuente** (no toca la seguridad): `python
  src/setup_wizard.py`, `python src/ghost_terminal.py`, etc. El intérprete de
  Python sí está firmado, así que SAC no se interpone.
- **Desactivar Smart App Control**: *Seguridad de Windows → Control de
  aplicaciones y explorador → Control inteligente de aplicaciones → Desactivar*.
  **Es irreversible**: una vez apagado, solo se vuelve a activar reinstalando
  Windows. Piénsalo antes.
