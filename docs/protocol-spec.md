# TokenTorrent: Especificación del protocolo

Este documento describe el protocolo **vigente**. El diseño original de la Fase 1
se conserva al final como apéndice histórico.

> [!NOTE]
> Lo que este protocolo **no** ofrece: privacidad diferencial formal, y una
> prueba criptográfica completa de ejecución (ver §5, Proof of Compute, para lo
> que sí garantiza y lo que no).

## 1. Transporte

Cada nodo levanta un servidor HTTP asíncrono (`aiohttp`) con cuatro endpoints:

| Ruta | Método | Propósito |
|---|---|---|
| `/ping` | GET | Salud e identidad del nodo (`node_id`, capas, arquitectura, si tiene TLS) |
| `/forward` | POST | Recibe un sobre, procesa sus capas y lo reenvía. Responde `202` de inmediato |
| `/callback` | POST | El cliente recibe aquí los logits finales o una notificación de error |
| `/attest` | POST | Reto de Proof of Compute (§5) |

El procesamiento es asíncrono: `/forward` responde `202 Accepted` y reenvía en
una tarea de fondo. Por eso un error aguas abajo **no** aparece como estado HTTP
en el emisor; se propaga con el sobre de error de §4.

### 1.1. TLS y mTLS

La identidad de un nodo es su **certificado X.509**, no su IP. Como las IP
domésticas son dinámicas, el certificado hoja lleva el `node_id` en un SAN
`dNSName` con el formato `<node_id>.node.tokentorrent`, y el cliente conecta a la
IP pasando `server_hostname=<node_id>.node.tokentorrent`. Así la verificación de
hostname de OpenSSL valida la identidad del nodo con independencia de la IP.

* Claves EC P-256; TLS 1.2 como mínimo.
* Con mTLS (`--tls-ca` sin `--no-client-cert`), OpenSSL rechaza en el handshake
  a quien no presente un certificado de la CA del enjambre.
* La PKI se genera con `src/gen_certs.py` (CA, certificados de nodo y clave del
  tracker).

### 1.2. Autenticación

Un nodo acepta una petición si se cumple **una** de estas condiciones:

1. presenta un certificado de cliente válido de la CA del enjambre (mTLS);
2. envía la cabecera `X-Auth-Token` que coincide con el secreto compartido;
3. no hay token configurado y el origen es loopback.

## 2. Serialización

`msgpack` + `zlib`. Los tensores se codifican como
`{'_is_tensor': True, shape, dtype, data, secret_sauce}` y pueden anidarse
dentro de diccionarios y listas. `secret_sauce` degrada float32/bfloat16 a
float16 antes de enviar (~2x menos bytes, con pérdida).

La deserialización es una frontera de seguridad: lista blanca de dtypes, y
límites de elementos, dimensiones, longitud de bytes, tamaño comprimido y
descomprimido, y número de elementos de lista.

## 3. Sobres

### 3.1. Sobre v1 (heredado, encadenamiento estático)

```
{ "request_id": "uuid-v4", "payload": <tensor | {hidden_states, position_ids}> }
```

El destino lo decide cada worker con su `--next-node`, fijado al arrancar. El
tracker no puede reencaminar nada porque el camino no viaja con el tensor.

### 3.2. Sobre v2 (enrutamiento en origen)

```
{
  "v": 2,
  "request_id": "uuid-v4",
  "route": [ "https://a:8001/forward", {"url": "https://b:8002/forward", "node_id": "..."}, "https://cliente:8000/callback" ],
  "hop": 0,
  "route_exp": 1765000000,
  "route_sig": <bytes Ed25519>,
  "payload": <tensor | dict>
}
```

* El último elemento de `route` es siempre el `/callback` del cliente.
* El nodo que recibe con `hop = n` reenvía a `route[n+1]` con `hop = n+1`.
* Un salto puede ser una URL o `{"url", "node_id"}`; el `node_id` se usa como
  SNI para validar la identidad del destino.
* Los workers dejan de necesitar `--next-node`: son apátridas respecto de la
  topología, que es lo que permite al tracker reasignar el camino.

### 3.3. Política de aceptación de rutas

Una lista de URLs que llega por la red es un vector de **SSRF y amplificación**:
un cliente malicioso podría hacer que N nodos del enjambre POSTeen contra la
víctima que elija. Un nodo solo acepta una ruta si:

1. viene firmada por el tracker (Ed25519) sobre la forma **normalizada** de la
   ruta, no ha caducado (`route_exp`), y el nodo tiene la clave pública
   (`--tracker-key`); **o**
2. todos sus saltos son loopback (desarrollo local); **o**
3. el nodo arrancó con `--allow-unsigned-routes` (asume el riesgo).

Además: máximo 16 saltos, solo esquemas `http`/`https`, sin credenciales en la
URL, y la ruta solo puede terminar en `/forward` o `/callback`.

## 4. Errores y tolerancia a fallos

Cuando un nodo no puede entregar al salto siguiente, envía al `/callback` del
cliente (último elemento de la ruta) un sobre de error:

```
{ "v": 2, "request_id": "...", "error": {"message": "...", "hop": 1, "node_id": "..."} }
```

El cliente falla rápido en vez de esperar el timeout de 30 s, marca al nodo como
caído, pide una ruta nueva que lo excluya y **reintenta el token** (hasta 3
intentos). El reintento es barato porque cada token recalcula el prompt completo:
no hay caché KV compartida entre saltos.

## 5. Proof of Compute

### Qué garantiza y qué no

Una prueba criptográfica de que un stack de capas transformer se ejecutó
correctamente (zkML) está hoy fuera de presupuesto para un modelo de 0.5B en CPU
doméstica. Lo implementado es **verificación probabilística con responsabilidad
atribuible**:

* **Detecta**: nodos que devuelven ruido, que se saltan capas o que alojan otro
  modelo.
* **No detecta**: un nodo con los pesos que calcula honestamente pero lento; un
  Sybil que pasa la atestación y luego hace trampa a tasa baja; la colusión
  entre auditado y auditor.

### 5.1. Atestación (`POST /attest`)

Petición: `{"seed": "<aleatoria>", "seq_len": 8, "hidden_size": 896}`.

El nodo deriva de la semilla una entrada canónica determinista, la procesa con
un solo hilo y algoritmos deterministas, y responde con la salida, su
`digest` SHA-256 y una **firma ECDSA** hecha con la clave privada de su
certificado sobre `"tokentorrent-attest-v1|<node_id>|<seed>|<digest>"`.

La semilla debe ser impredecible: si el nodo la conociera de antemano podría
precalcular la respuesta sin alojar las capas. La firma liga la salida a la
identidad y a la semilla, así que no es reutilizable y una salida incorrecta es
evidencia no repudiable.

Verificador: `src/attest.py` carga localmente el mismo rango de capas, calcula
la referencia y compara.

### 5.2. Comparación numérica

Nunca bit a bit: el downcast a float16 del `secret_sauce`, el orden de reducción
entre hilos y la versión de PyTorch hacen imposible la igualdad exacta. Se
compara la **distancia L2 relativa** contra `DEFAULT_EPSILON` (2e-2). Las
respuestas de `/attest` se serializan con `secret_sauce` desactivado para que
los bytes firmados sean exactamente los verificados.

### 5.3. Auditoría por muestreo

El cliente, con probabilidad `p` (`--audit-probability`), repite un paso por una
ruta alternativa que excluya a los nodos ya usados y compara. La probabilidad de
detectar a un tramposo en `n` pasos es `1 - (1 - p)^n`. Una divergencia se
reporta al tracker (`/report`) y el nodo queda excluido de las rutas siguientes.

Sin identidad criptográfica (§1.1) esto no sirve de nada: un `node_id` que es un
UUID efímero se rota reiniciando el proceso.

## 6. Tracker

`src/tracker.py` es una implementación de referencia ejecutable del contrato
(el tracker de producción, JARVIS, vive fuera de este repositorio). Sirve para
levantar un enjambre completo en local:

```bash
python src/tracker.py --port 5000 --signing-key certs/tracker.key
```

| Ruta | Método | Propósito |
|---|---|---|
| `/register` | POST | Latido del nodo (cada **15 s**) con capas, `is_last`, recursos donados y certificado |
| `/plan` | GET | Devuelve `{route, exp, signature}`: el camino completo firmado |
| `/route` | GET | Heredado: lista de nodos sueltos |
| `/report` | POST | Recibe divergencias detectadas en auditoría |
| `/status` | GET | Vista legible del enjambre |

### 6.1. Nodos fantasma

Un PC apagado tirando del cable deja de latir sin avisar. El tracker expulsa a
todo nodo sin latido en **45 segundos**, que son exactamente tres latidos
perdidos: suficiente para tolerar un fallo de red puntual sin dejar de enrutar
hacia nodos vivos. El nodo, por su parte, reintenta el registro con espera
exponencial (15 s → 30 s → … → 300 s) y reinicia sus conexiones cuando detecta
que la red se cayó, porque el pool de sockets conserva conexiones muertas.

### 6.2. El callback va dentro de la firma

`/plan` **exige** el parámetro `callback` con la URL del cliente, y firma la
ruta completa incluyéndolo. No es un detalle de comodidad: si el cliente
pudiera añadir el último salto después de la firma, podría apuntar al destino
que quisiera y convertir a todo el enjambre en un amplificador contra esa
víctima.

```
GET /plan?model_arch=llama/qwen&start_layer=8&callback=https://cliente:8000/callback
```

El tracker solo devuelve rutas cuya cadena de capas es contigua y termina en un
nodo con `is_last`; una cadena incompleta nunca produciría logits, así que se
responde `404` en vez de dejar al cliente esperando un callback que no llegará.

### 6.3. Limitador de peticiones

Todos los endpoints —los del tracker y los del nodo— aplican un *token bucket*
por identidad (certificado si lo hay, IP si no), implementado en
`src/ratelimit.py`. Al exceder el ritmo se responde `429` con `Retry-After`.
El diccionario de cubos tiene un tope de clientes: indexar por IP sin límite
sería, en sí mismo, una vía de agotamiento de memoria.

| Endpoint | Ritmo | Ráfaga |
|---|---|---|
| Nodo `/forward`, `/ping`, `/callback` | 25/s | 100 |
| Nodo `/attest` | 0,2/s | 3 |
| Tracker `/register` | 2/s | 10 |
| Tracker `/plan`, `/route` | 5/s | 30 |
| Tracker `/report` | 0,5/s | 5 |

`/attest` tiene un cupo propio y mucho más estricto porque cada llamada ejecuta
inferencia real: es el endpoint más caro del nodo.

---

# Apéndice: especificación histórica (Fase 1)

> [!WARNING]
> **Documento histórico.** Este esquema de payload (`tensor_data`, `shape` y
> `client_callback` en el cuerpo) **no** corresponde a la implementación actual;
> el formato real está en §2 y §3.

Durante la Fase 1 se usó una topología de anillo/pipeline estática. El
descubrimiento era manual: al iniciar un nodo se le pasaba la IP y el puerto del
siguiente. El último nodo devolvía el resultado al cliente original.

```json
{
  "request_id": "uuid-v4",
  "tensor_data": "<bytes>",
  "shape": [1, 128],
  "dtype": "float32",
  "client_callback": "http://192.168.1.10:8000/callback"
}
```

El manejo de errores consistía en registrar el fallo y abortar el pipeline, sin
notificar al cliente: de ahí el sobre de error de §4.
