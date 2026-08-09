# Plan técnico — TLS, PoC, Tracker, VM

> **Estado (2026-08-09): las tareas 1, 2 y 3 están implementadas**, junto con los
> prerrequisitos de código de la 4. Este documento se conserva como registro de
> diseño y justificación de las decisiones; el protocolo resultante está descrito
> en [`protocol-spec.md`](protocol-spec.md).
>
> | Tarea | Estado | Dónde |
> |---|---|---|
> | 1. TLS/mTLS | Implementado | `src/tls_utils.py`, `src/gen_certs.py`, flags `--tls-*` |
> | 2. Proof of Compute | Implementado (atestación firmada + auditoría por muestreo; **no** zkML, ver §2) | `src/pow_utils.py`, `src/attest.py`, `POST /attest` |
> | 3. Orquestación del tracker | Implementado del lado del enjambre; falta el servicio JARVIS (`/plan`, `/report`) | `src/routing.py`, sobre v2 |
> | 4. Pruebas en VM | Prerrequisitos de código hechos (logging, propagación de entorno); la VM sigue pendiente | §4 |

Cada sección indica el estado del código previo, el diseño aplicado, los ficheros
afectados y los riesgos.

> **Ronda v2.0 (pulido y usabilidad), también aplicada:** configuración
> persistente (`src/config.py`), limitador de peticiones (`src/ratelimit.py`),
> tracker de referencia con expulsión de nodos fantasma a los 45 s
> (`src/tracker.py`), QoS con límites de CPU/RAM elegidos por el usuario,
> fallback de puerto, watchdog de red, icono de bandeja y panel de Ajustes en la
> interfaz, ventana de instalación redimensionable con detección de puertos, y
> desinstalador con kill-switch. Detalle en el README y en `protocol-spec.md` §6.

## Orden recomendado y dependencias

```
1. TLS/mTLS  ──►  2. Proof of Compute      (PoC necesita identidad no falsificable)
      │
      └────────►  3. Orquestación Tracker  (rutas firmadas necesitan una PKI)
                        │
                        └──►  4. Pruebas en VM   (última: valida el binario final)
```

Las dos dependencias duras:

- **PoC depende de TLS.** Hoy `P2PNode.node_id` es un `uuid.uuid4()` generado en cada arranque y `self.public_key` es `None` ([p2p_node.py:44](../src/p2p_node.py#L44)). Sin identidad criptográfica, un nodo que falle una auditoría se reinicia con un `node_id` nuevo y vuelve limpio al enjambre. Cualquier sistema de reputación o *slashing* es papel mojado hasta que la identidad sea la clave privada de un certificado.
- **La orquestación desde el tracker exige cambiar el sobre del protocolo.** La ruta hoy está en los argumentos CLI de cada worker (`--next-node`), fijada al arrancar el proceso. El tracker no puede reencaminar nada porque el camino no viaja con el tensor.

---

## 1. TLS / mTLS

### Estado actual

- `web.TCPSite(runner, self.host, self.port)` sin `ssl_context` ([p2p_node.py:318](../src/p2p_node.py#L318)); igual en `AgenticChat.setup()` ([chat_agent.py:157](../src/chat_agent.py#L157)). Todo el tráfico es HTTP plano.
- `_forward_to_next` ya **acepta** `https` en la validación de esquema ([p2p_node.py:178](../src/p2p_node.py#L178)), pero crea un `ClientSession()` desnudo por llamada, sin contexto SSL ni verificación configurable.
- La autenticación es un *bearer token* compartido comparado con `hmac.compare_digest` ([p2p_node.py:74-81](../src/p2p_node.py#L74)). Sobre HTTP plano ese token viaja en claro en cada petición: cualquiera en la ruta lo captura y se hace pasar por un nodo del enjambre.

### Diseño propuesto: CA privada del enjambre + mTLS

Descartadas: ACME/CA pública (los nodos domésticos no tienen nombre DNS estable) y *self-signed* sin ancla (no distingue nodo legítimo de atacante).

**El problema de identidad y su solución.** Los nodos se alcanzan por IP dinámica, así que un certificado con SAN de IP caduca en cuanto el ISP rota la dirección. Propuesta: emitir el certificado hoja con el `node_id` como `SAN:dNSName` (ej. `a3f1....node.tokentorrent`), publicar ese `node_id` en el tracker, y conectar a la IP pero pasando `server_hostname=<node_id>`. Así la verificación de hostname de OpenSSL valida **identidad de nodo**, desacoplada de la IP. Es el punto central del diseño.

**Módulo nuevo `src/tls_utils.py`:**

```
build_server_ssl_context(certfile, keyfile, cafile, require_client_cert=True) -> ssl.SSLContext
build_client_ssl_context(certfile, keyfile, cafile) -> ssl.SSLContext
peer_node_id(request) -> str | None      # extrae el SAN del cert de cliente
```

- `minimum_version = ssl.TLSVersion.TLSv1_2` (preferir 1.3), `verify_mode = ssl.CERT_REQUIRED`, `load_verify_locations(cafile)`, `load_cert_chain(certfile, keyfile)` en ambos lados.
- En servidor, el cert del par se obtiene con `request.transport.get_extra_info('ssl_object').getpeercert()`.

**Cambios en `P2PNode`:**

- `__init__` acepta `tls_cert`, `tls_key`, `tls_ca`, `require_client_cert`; `start()` pasa `ssl_context=` a `TCPSite`.
- **Refactor obligatorio del cliente HTTP**: hoy `_forward_to_next` abre un `ClientSession` nuevo por cada reenvío ([p2p_node.py:194](../src/p2p_node.py#L194)). Con TLS eso es un *handshake* completo por **hop y por token** — inaceptable. Crear una única `ClientSession` con `TCPConnector(ssl=client_ctx)` en `start()`, guardarla en `self._client_session` y cerrarla en `_cleanup`. Esta mejora ya merece la pena por sí sola aunque TLS se retrase.
- `_check_auth` pasa a dos niveles: **autenticación** = cert firmado por la CA del enjambre; **autorización** = `node_id` del cert contra una allowlist opcional. El token compartido se mantiene como compatibilidad hacia atrás y segundo factor, no como mecanismo principal.
- El heartbeat debe anunciar el esquema (`https`) y el `node_id` del certificado, no el UUID efímero ([p2p_node.py:245-254](../src/p2p_node.py#L245)).

**CLI (`worker.py`, `chat_agent.py`):** `--tls-cert`, `--tls-key`, `--tls-ca`, `--require-client-cert`. Endurecer la comprobación existente de [worker.py:141](../src/worker.py#L141): hoy escuchar fuera de localhost solo exige token; debe exigir **TLS**. Mismo criterio para `--enable-upnp`.

**Aprovisionamiento de certificados.** Es la parte no trivial. El instalador hoy genera `secrets.token_urlsafe(32)` ([setup_wizard.py:213](../src/setup_wizard.py#L213)) y lo escribe en `HKCU\Environment`. Dos opciones:

- *(recomendada, medio plazo)* Flujo CSR contra el tracker: el instalador genera par de claves, manda CSR, el tracker firma como CA. Requiere endpoint nuevo en JARVIS y una política anti-Sybil para decidir a quién firma.
- *(interina)* Autofirmado local + *pinning* TOFU: el nodo registra el hash SPKI en el heartbeat y los clientes lo fijan desde la respuesta del tracker. Más barato, pero convierte al tracker en la raíz de confianza de facto.

Ambas añaden **dependencia nueva `cryptography`** a `requirements.txt` y al empaquetado PyInstaller.

**Pruebas.** Extender `tests/test_protocol.py`: *fixture* que genere una CA efímera + dos hojas en `tmp_path`, y tres casos — pipeline mTLS correcto, cliente en claro rechazado, cliente con cert de otra CA rechazado.

**Riesgos.** Rotación/caducidad en un worker de autoarranque que puede correr meses sin supervisión (hace falta renovación automática); tamaño y correcta inclusión de OpenSSL/`certifi` en el binario PyInstaller; depuración de fallos de handshake con `--noconsole` (ver §4).

---

## 2. Proof of Compute

### Evaluación honesta previa

Una prueba criptográfica de que un stack de capas transformer se ejecutó correctamente (zkML, pruebas succintas sobre la inferencia) está hoy varios órdenes de magnitud fuera de presupuesto para un modelo de 0.5B en CPU doméstica. **Lo alcanzable es verificación probabilística + responsabilidad atribuible, no una prueba matemática.** Conviene que el README y el spec lo digan con esas palabras para no prometer de más.

### Obstáculo técnico concreto: el determinismo

`_pack_single_tensor` degrada float32/bfloat16 a **float16** cuando `use_secret_sauce=True`, que es el valor por defecto ([tensor_utils.py:26-38](../src/tensor_utils.py#L26)). Esa pérdida, sumada a variación de hardware, orden de reducción y versión de PyTorch, hace **imposible la comparación bit a bit**. Toda verificación debe:

1. ejecutarse con `use_secret_sauce=False`,
2. comparar con tolerancia (L2 relativa o similitud coseno bajo ε), no con igualdad,
3. y ε debe fijarse **empíricamente antes de escribir el verificador** — es un experimento previo, no un número inventado. Ampliar `tests/test_phase2_5_benchmark.py` para medir la divergencia de la misma entrada entre fp32/fp16/bf16 y entre máquinas distintas.

### Mecanismos, por orden de coste/beneficio

**(a) Reto de atestación en el registro — barato, alto valor.**
Endpoint nuevo `POST /attest`: el tracker manda una entrada canónica conocida para el par (modelo, rango de capas); el nodo devuelve la salida determinista **firmada con la clave privada de su certificado**. El tracker compara contra una referencia dentro de ε. Detecta de inmediato al que devuelve ruido y al que se salta capas (saltarse una capa desvía la salida mucho más allá de cualquier ε razonable). La firma hace la evidencia no repudiable, que es lo que habilita el *slashing*.

**(b) Auditoría por muestreo en caliente.**
El cliente, con probabilidad *p*, duplica un paso de token hacia un segundo nodo con el mismo rango de capas y compara. Discrepancia → reporte al tracker. La probabilidad de detección de un tramposo que engaña en *n* pasos es `1-(1-p)^n`; *p* se elige por el coste extra de cómputo asumible.

**(c) Compromiso Merkle + reto de subconjunto — fase posterior.**
El nodo devuelve la salida más un compromiso Merkle sobre las activaciones intermedias por capa; el verificador reta *k* capas al azar y el nodo revela esas rebanadas con su camino Merkle. Solo puede verificarlo quien tenga los pesos de esas capas (otro nodo del mismo rango). Coste: mantener intermedios en memoria y un endpoint `/challenge`.

**(d) Reputación y penalización.**
Sobre (a)/(b): el tracker acumula auditorías fallidas por `node_id` **del certificado** y degrada o expulsa. Sin la tarea 1, inútil.

**Ruta recomendada:** (a) + (d) primero, (b) después, (c) solo si hay adversario económico real.

**Lo que este esquema NO cubre** — decirlo explícitamente en el spec: un nodo con los pesos que calcula honestamente pero muy lento; un Sybil que pasa la atestación y luego hace trampa a tasa baja; colusión entre el nodo auditado y el auditor.

---

## 3. Coordinación del pipeline desde el Tracker

### Estado actual

- `chat_agent` hace `GET /route?model_arch=…`, coge `nodes[0]` y construye una única URL ([chat_agent.py:225-230](../src/chat_agent.py#L225)). No hay noción de camino.
- El resto del camino está congelado en los `--next-node` de cada worker.
- Un fallo intermedio se registra y se descarta ([p2p_node.py:200-203](../src/p2p_node.py#L200)); el cliente no se entera y se come el timeout de 30 s ([chat_agent.py:267](../src/chat_agent.py#L267)).

### Diseño propuesto: enrutamiento en origen con ruta firmada

El sobre pasa a llevar el camino completo:

```
{ v: 2, request_id, route: [url_hop1, url_hop2, …, url_callback], hop: 0, payload }
```

`handle_forward` lee `route[hop]`, incrementa `hop` y reenvía; el último salto entrega al callback del cliente. Los workers dejan de necesitar `--next-node` y se vuelven apátridas respecto de la topología, que es lo que permite al tracker reasignar.

**Cambios necesarios, en orden:**

1. **`tensor_utils`**: `_pack_value` **rechaza listas** hoy ([tensor_utils.py:80-82](../src/tensor_utils.py#L80)). Hay que admitir `list` con un tope de longitud (ruta máx. ~16 saltos) y reflejarlo en `_unpack_value`.
2. **`handle_forward`** exige hoy `set(incoming) == {'request_id','payload'}` ([p2p_node.py:114](../src/p2p_node.py#L114)). Versionar el sobre con el campo `v` y aceptar ambos formatos durante la transición.
3. **Seguridad — crítico.** Una lista de URLs que llega por la red es un vector de **SSRF y amplificación**: un cliente malicioso puede hacer que N nodos del enjambre POSTeen a la víctima que elija. Mitigaciones obligatorias, no opcionales: la ruta la **firma el tracker** y cada nodo verifica la firma antes de reenviar; tope de saltos; esquema/puertos restringidos; prohibir destinos en rangos privados salvo configuración explícita.
4. **Caché KV**: hoy la clave es el sha256 del cuerpo crudo ([p2p_node.py:104](../src/p2p_node.py#L104)). Con `hop` dentro del cuerpo, la clave cambia en cada salto y la caché deja de acertar en silencio. Rehacer la clave sobre el hash del `payload` únicamente.
5. **Tracker (servicio externo JARVIS)**: endpoint `/plan` que devuelva una ruta firmada que cubra las capas 8..N para un `model_arch`. El heartbeat actual cada 120 s ([p2p_node.py:264](../src/p2p_node.py#L264)) es **demasiado lento para failover**: hace falta TTL corto o sondeo activo de `/ping` por parte del cliente.
6. **Fallo en mitad del pipeline**: el nodo que no puede reenviar debe emitir un callback de error a `route[-1]` con el `request_id` y el salto fallido, para que el cliente falle rápido en vez de esperar 30 s. El cliente pide ruta nueva excluyendo al nodo caído y **reintenta el token**. El reintento es barato precisamente porque hoy no hay caché KV entre saltos y cada token recalcula el prompt completo (que es también la razón de que sea lento).
7. **`chat_agent._generate_response`**: sustituir `self.primary_node_url` por un objeto ruta + `_refresh_route()`; el camino de error actual (`self.primary_node_url = None; break`, [chat_agent.py:259-264](../src/chat_agent.py#L259)) pasa a replanificar y reintentar con tope de intentos.

**Optimización a considerar:** asignar la pipeline por *sesión* (N tokens) en vez de por token, para no meter un round-trip al tracker en cada token.

---

## 4. Pruebas en VM Windows

No es solo "probar los .exe": hay cuatro cosas en el código que harán fallar la prueba o la harán ininterpretable si no se arreglan **antes** de montar la VM.

### Prerrequisitos antes de la primera VM

- **Logging a fichero en el worker.** `p2p_node.exe` se compila con `--noconsole` ([build.py](../build.py)). Si `worker.py` aborta por `parser.error` (por ejemplo, token ausente) sale con código 2 **sin ninguna traza visible**. Añadir `FileHandler` a `%LOCALAPPDATA%\GhostTerminal\worker.log` antes de tocar la VM; si no, cualquier fallo será indistinguible de "no arranca".
- **Propagación de la variable de entorno.** `register_worker_autostart` escribe `TOKENTORRENT_AUTH_TOKEN` en `HKCU\Environment` ([setup_wizard.py:322-323](../src/setup_wizard.py#L322)) sin difundir `WM_SETTINGCHANGE`. La entrada `Run` la recogerá en el siguiente inicio de sesión, pero hay que **verificarlo explícitamente en la VM**, porque el modo de fallo es el silencioso del punto anterior.
- ~~**`build.py` borra los `.spec` versionados.**~~ **Resuelto:** `ghost_bundle.spec` es ahora código fuente versionado y `build.py` lo respeta al limpiar; el resto de `*.spec` se siguen borrando por ser artefactos.
- **Autoarranque con `--enable-upnp --host 0.0.0.0`** ([setup_wizard.py:324](../src/setup_wizard.py#L324)): abrir puerto al exterior por defecto en la instalación es una decisión de producto que conviene revisar antes de distribuir binarios, y que TLS (tarea 1) debería condicionar.

### Antivirus: el riesgo real

La combinación acumula casi todas las heurísticas clásicas de falso positivo:

| Factor | Por qué puntúa |
|---|---|
| PyInstaller `--onefile` sin firmar | autoextracción a temp, patrón compartido con *packers* |
| módulo `keyboard` (hotkey global) | hook de teclado ⇒ heurística de *keylogger* |
| clave de registro `Run` | persistencia |
| `--uac-admin` en el instalador | elevación |
| apertura de puerto por UPnP | comportamiento de red no solicitado |

Mitigaciones, de mayor a menor efecto: **certificado de firma de código** (la única solución de fondo, en marcha vía [SignPath](firma-codigo.md)); ~~pasar a `--onedir`~~ **hecho** (`ghost_bundle.spec`: ya no hay autoextracción en `%TEMP%`); enviar los binarios a Microsoft como falso positivo; sustituir `keyboard` por una API de hotkey menos marcada.

### Matriz de pruebas

- Win10 y Win11 limpios, sin Python, Defender activo y actualizado.
- Usuario estándar **y** administrador (la instalación escribe en HKCU y `LOCALAPPDATA`, pero el instalador pide `--uac-admin`).
- **Primer arranque sin conexión**: `transformers` descargará el modelo de HuggingFace en el primer uso — `--collect-all transformers` no empaqueta pesos. Comprobar el mensaje de error y que la caché HF sea escribible.
- Tiempo de arranque: con `--onedir` ya no hay extracción por lanzamiento, pero conviene medirlo igualmente en la VM (la carga de `torch_cpu.dll` sigue siendo pesada).
- **Desinstalación**: que `uninstaller.exe` retire las dos claves `Run`, la entrada de `PATH` y `TOKENTORRENT_AUTH_TOKEN`.
- Reinstalación sobre una instalación previa (el instalador hace `taskkill` de los dos ejecutables; verificar que no quedan ficheros bloqueados).

---

## Impacto en documentación

Al cerrar las tareas 1–3 hay que actualizar: los avisos de `README.md` sobre ausencia de TLS y de Proof of Compute, el bloque `[!WARNING]` de `docs/protocol-spec.md`, y la sección "Security posture" de `CLAUDE.md`. El spec necesita además una sección nueva con el formato de sobre v2.
