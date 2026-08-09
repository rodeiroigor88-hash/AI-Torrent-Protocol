# TokenTorrent

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![Status](https://img.shields.io/badge/Status-Beta-brightgreen)

**TokenTorrent** es un ecosistema descentralizado (P2P) de código abierto diseñado para democratizar la Inteligencia Artificial. Al igual que BitTorrent revolucionó el intercambio de archivos, este protocolo permite a múltiples usuarios de bajos recursos unir su memoria RAM y CPU para ejecutar Modelos de Lenguaje Masivos (LLMs) de forma colectiva y gratuita.

Diseñado por [**rodeiroigor88**](https://github.com/rodeiroigor88).

## 🚀 ¿Por qué TokenTorrent?

Actualmente, ejecutar modelos potentes como Llama 3 o Qwen requiere GPUs inalcanzables para la mayoría. **TokenTorrent** resuelve este problema usando *Pipeline Parallelism Dinámico*:
1. El modelo de IA se fragmenta en "Capas" (Layers).
2. Cada ordenador de la red (incluso aquellos con solo 500MB de RAM) aloja un par de capas.
3. Los tensores viajan de ordenador en ordenador a través de internet procesando la información.
4. El usuario final recibe la respuesta como si estuviera chateando con un superordenador en la nube.

## ✨ Características Principales

* **100% Descentralizado:** Sin APIs corporativas. El control vuelve a la comunidad.
* **Tolerancia de Bajos Recursos:** Los nodos (Workers) pueden funcionar en PCs muy antiguos haciendo un casting automático a `bfloat16`.
* **Protocolo binario limitado:** Los tensores usan `msgpack` + `zlib`, con validación estricta y límites de tamaño. La compresión no equivale a cifrado.
* **TLS / mTLS:** El tráfico nodo-a-nodo puede cifrarse con certificados de una CA privada del enjambre. La identidad de un nodo es su certificado, no su IP.
* **Proof of Compute:** Un reto firmado (`/attest`) demuestra que un nodo ejecuta realmente las capas que dice alojar, más auditoría por muestreo durante la generación.
* **Pipeline orquestable:** El camino del tensor viaja dentro del sobre y puede venir firmado por el tracker, que reasigna la ruta cuando un nodo cae.
* **Arquitectura asíncrona:** La red usa `aiohttp` y deriva la inferencia a un hilo controlado para no bloquear el event loop.

## 🛠 Instalación y Uso

### 1. Clonar e Instalar
```bash
git clone https://github.com/rodeiroigor88-hash/TokenTorrent.git
cd TokenTorrent
pip install -r requirements.txt
```

### 2. Iniciar un Worker (Semilla)
Para donar recursos y alojar capas específicas del modelo (ej. de la 8 a la 15):
```bash
python src/worker.py --layers 8-15 --port 8001
```

### 3. Iniciar el Chat Agent
Para interactuar con la red P2P a través de la terminal:
```bash
python src/chat_agent.py
```

Si el agente detecta un bloque de código con un comentario de nombre de archivo (ej. `# script.py`) en la respuesta del modelo, preguntará `(y/n)` antes de escribirlo a disco.

### 4. Permitir conexiones remotas (con TLS)
Por defecto, el worker escucha únicamente en `127.0.0.1`. **Escuchar fuera de localhost exige TLS.** Primero se crea la PKI del enjambre:

```bash
python src/gen_certs.py init-ca --out certs
```
```bash
python src/gen_certs.py node --ca-dir certs --name worker1 --ip 192.168.1.40
```
```bash
python src/gen_certs.py node --ca-dir certs --name cliente
```

Cada nodo necesita su par `.crt`/`.key` y el `ca.crt` común (la `ca.key` no se distribuye):

```bash
python src/worker.py --host 0.0.0.0 --port 8001 --layers 8-15 --tls-cert certs/worker1.crt --tls-key certs/worker1.key --tls-ca certs/ca.crt
```
```bash
python src/chat_agent.py --tls-cert certs/cliente.crt --tls-key certs/cliente.key --tls-ca certs/ca.crt --route https://192.168.1.40:8001/forward#<node_id>
```

Con mTLS el certificado sustituye al token: quien no lo presente cae en el handshake. Si quieres exponer el nodo **sin** cifrar (el token y los tensores viajan en claro, interceptables), hay que asumirlo explícitamente con `--insecure-no-tls` y un `--auth-token`.

UPnP y el registro en el tracker siguen desactivados por defecto (`--enable-upnp`, `--enable-tracker`). Este prototipo no implementa una garantía formal de privacidad diferencial.

### 5. Verificar que un nodo calcula de verdad
El reto de atestación carga localmente el mismo rango de capas y compara la salida del nodo con la referencia:

```bash
python src/attest.py --node https://192.168.1.40:8001 --layers 8-15 --tls-ca certs/ca.crt --tls-cert certs/cliente.crt --tls-key certs/cliente.key
```

Durante la generación, el cliente puede auditar pasos al azar por una ruta alternativa:

```bash
python src/chat_agent.py --audit-probability 0.05
```

Detecta a quien devuelve ruido, se salta capas o aloja otro modelo. **No** detecta a quien calcula honestamente pero lento, ni a un Sybil que hace trampa a tasa baja. Ver `docs/protocol-spec.md` §5.

### 6. Enjambre local completo (con tracker propio)
El tracker de referencia permite probarlo todo sin depender del servidor de JARVIS:

```bash
python src/tracker.py --port 5000 --signing-key certs/tracker.key
```

Los workers validan las rutas firmadas con `--tracker-key certs/tracker.pub`, y el cliente apunta al tracker local con `--tracker-url http://127.0.0.1:5000` (o desde **Ajustes** en la interfaz). `GET /status` da una vista legible del enjambre.

### 7. Cuántos recursos donas
El worker limita lo que cede y se auto-pausa si el PC se satura:

```bash
python src/worker.py --port 8001 --layers 8-15 --cpu-cores 2 --max-ram-percent 25
```

`--cpu-cores 0` (por defecto) dona la mitad de los núcleos. Estos valores, el atajo de teclado y la URL del tracker se guardan en `%LOCALAPPDATA%\GhostTerminal\config.json` y se editan desde el panel de **Ajustes**, sin recompilar.

## 🖥 La interfaz (Ghost Terminal)

* **Icono en la bandeja del sistema:** un fantasmita verde junto al reloj. Doble clic abre el terminal; con clic derecho hay menú de *Ajustes* y *Salir*. Sin él, quien olvidara el atajo no tenía forma de recuperar la aplicación.
* **Panel de Ajustes:** cambia el atajo de teclado (por defecto `F12`), sus alternativas y la URL del tracker. Los cambios se aplican en caliente.
* **Reconexión:** si el tracker se cae, el cliente reintenta con espera exponencial mostrando *"Reconectando con la red P2P…"* en vez de rendirse al primer intento.

## 🧱 Estructura de la Red P2P (Fase 3)

* `src/p2p_node.py`: Gestión de red asíncrona basada en AIOHTTP para el tráfico de tensores.
* `src/worker.py`: Lógica de inferencia descentralizada (carga selectiva de capas HF Transformers).
* `src/chat_agent.py`: Cerebro local del cliente con capacidades de ejecución autónoma (Agents).
* `src/tensor_utils.py`: Optimización, compresión y serialización de tensores PyTorch.
* `src/routing.py`: Enrutamiento en origen, firma de rutas y defensa contra SSRF.
* `src/tls_utils.py`: PKI del enjambre y contextos TLS/mTLS.
* `src/pow_utils.py`: Reto determinista y comparación con tolerancia (Proof of Compute).
* `src/tracker.py`: Tracker de referencia (planificación firmada, expulsión de nodos fantasma, rate limit).
* `src/ratelimit.py`: Token bucket compartido por nodos y tracker.
* `src/config.py`: Configuración persistente (atajo, tracker, puertos, recursos donados).
* `src/gen_certs.py` / `src/attest.py`: Herramientas de línea de comandos para la PKI y la verificación de nodos.

## ⬇️ Descargas y firma de código

Las versiones publicadas están en la [página de Releases](https://github.com/rodeiroigor88-hash/TokenTorrent/releases). Descarga el `.zip`, descomprímelo y ejecuta `Ghost Terminal Setup.exe`; el asistente instala la carpeta `GhostTerminal` que viene a su lado.

**Requisitos de espacio:** unos 290 MB de descarga y ~875 MB instalados. El peso viene de PyTorch (`torch_cpu.dll` son 291 MB por sí solos), que va empaquetado para que no tengas que instalar Python. Los tres ejecutables comparten un único runtime en `_internal`, así que PyTorch se incluye **una sola vez**: antes se duplicaba y el instalador pesaba 602 MB.

Además hace falta conexión en el primer arranque: el modelo se descarga de HuggingFace, no viene dentro del instalador.

Los binarios de Windows (`Ghost Terminal Setup.exe`, `ghost_terminal.exe`, `p2p_node.exe` y `uninstaller.exe`) se firman digitalmente de forma gratuita gracias a la [**SignPath Foundation**](https://signpath.org/), con un certificado proporcionado por [**SignPath.io**](https://signpath.io/). Sin esa firma, Windows bloquearía la instalación mediante Control inteligente de aplicaciones (*Smart App Control*).

Puedes verificar la firma de cualquier binario descargado con:

```bash
python sign.py --verify "Ghost Terminal Setup.exe"
```

El proceso de firma está documentado en [docs/firma-codigo.md](docs/firma-codigo.md).

## 🔒 Privacidad

Este es un sistema P2P: para funcionar, parte de la información sale necesariamente de tu equipo. Esto es exactamente lo que ocurre, y nada más:

* **Si ejecutas un Worker (donas recursos):** tu nodo envía al tracker su identificador, la **dirección IP y el puerto** por los que es alcanzable, el rango de capas que aloja, la RAM y los núcleos donados. Es imprescindible para que otros nodos puedan encontrarte. El registro en el tracker está **desactivado por defecto** (`--enable-tracker`).
* **Si usas el cliente (chateas):** tu texto se convierte en tensores (*hidden states*) que viajan a los nodos del enjambre para completar la inferencia. **Esos tensores derivan de tu conversación.** La compresión no es cifrado: usa TLS/mTLS para que el tráfico no sea legible en tránsito.
* **Lo que NO se envía nunca:** el contenido de tus archivos, tus credenciales, ni telemetría de uso. No hay analítica, ni publicidad, ni perfilado. No existe una base de datos central de usuarios.
* **Lo que se queda en tu equipo:** la configuración (`%LOCALAPPDATA%\GhostTerminal\config.json`), el log del worker y las claves privadas de tus certificados, que nunca se transmiten.

Al ser software de código abierto, todo lo anterior es verificable leyendo el código: el registro en el tracker está en `src/p2p_node.py` y el envío de tensores en `src/chat_agent.py`.

Este proyecto no ofrece garantía de privacidad diferencial formal: un nodo del enjambre procesa activaciones derivadas de tu texto. Tenlo en cuenta antes de enviar información sensible.

## 🤝 Contribuir

¡Las Pull Requests son bienvenidas! Si deseas contribuir al protocolo de la próxima generación, abre un *Issue* o envía un PR. Estamos buscando implementar un Tracker DHT y Load Balancing Dinámico.

## 📄 Licencia

Este proyecto está bajo la licencia Apache 2.0, que incluye una concesión expresa de patentes. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---
*Created with passion by rodeiroigor88.*
