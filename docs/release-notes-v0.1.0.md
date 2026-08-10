# Notas para la primera release (v0.1.0)

Pegar este texto en **Releases → Draft a new release** de GitHub.

* **Tag:** `v0.1.0`
* **Título:** `TokenTorrent v0.1.0 — primera versión pública`
* **Marcar "Set as a pre-release":** sí, mientras los binarios no estén firmados.

Publicar esta release **sin adjuntar los .exe** es una opción perfectamente
válida: GitHub adjunta automáticamente el código fuente en `.zip` y `.tar.gz`,
así que la página deja de estar vacía y cumple el requisito de la SignPath
Foundation ("a page where users can download your software", que además debe
mencionar que el proyecto usa SignPath) sin distribuir todavía ejecutables que
Smart App Control bloquearía en casa de cada usuario.

---

## Texto de la release

TokenTorrent reparte las capas de un LLM entre varios ordenadores y las ejecuta
en cadena sobre HTTP, igual que BitTorrent reparte un fichero: **BitTorrent
mueve bits, TokenTorrent mueve tokens**. Cada vuelta completa del pipeline
produce un token.

### Qué incluye esta versión

* Transporte P2P asíncrono con enrutamiento en origen: el camino viaja dentro
  del sobre y el tracker puede reencaminarlo cuando un nodo cae.
* TLS/mTLS con una PKI propia del enjambre. La identidad de un nodo es su
  certificado, no su IP.
* Proof of Compute: un reto firmado demuestra que un nodo ejecuta realmente las
  capas que dice alojar, más auditoría por muestreo durante la generación.
* Tracker de referencia, limitador de peticiones y control de recursos donados.
* Interfaz de escritorio (Ghost Terminal) con icono en la bandeja del sistema.

### Firma de código

Los binarios de Windows de este proyecto se firman digitalmente de forma
gratuita gracias a la [**SignPath Foundation**](https://signpath.org/), con un
certificado proporcionado por [**SignPath.io**](https://signpath.io/).

> Esta pre-release **no incluye ejecutables firmados todavía**: la solicitud a
> la SignPath Foundation está en revisión. Mientras tanto, el proyecto se
> ejecuta desde el código fuente (ver el README). Los binarios llegarán en
> cuanto la firma esté disponible, porque sin ella Windows los bloquea mediante
> Control inteligente de aplicaciones.

### Privacidad

TokenTorrent es un sistema P2P y parte de la información sale necesariamente
del equipo (dirección IP y puerto hacia el tracker, tensores derivados de la
conversación hacia los nodos). Está detallado en la sección **Privacidad** del
README. No hay telemetría, ni analítica, ni base de datos central de usuarios.

### Licencia

Apache 2.0.
