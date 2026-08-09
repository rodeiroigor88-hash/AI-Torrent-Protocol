# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI Torrent Protocol: a P2P swarm that runs an LLM by splitting its transformer layers across machines (pipeline parallelism over HTTP). The client holds embeddings + the first layers; each worker holds a contiguous layer range; the last worker holds the final norm + `lm_head` and posts logits back to the client's callback. Code, comments and logs are in Spanish — keep new ones in Spanish for consistency.

## Commands

Install:
```bash
pip install -r requirements.txt
```

Run a 2-worker swarm locally (each command in its own shell, worker order defines the pipeline):
```bash
python src/worker.py --port 8001 --layers 8-15 --next-node http://127.0.0.1:8002/forward
```
```bash
python src/worker.py --port 8002 --layers 16-23 --next-node http://127.0.0.1:8000/callback --is-last
```
```bash
python src/chat_agent.py --port 8000 --next-node http://127.0.0.1:8001/forward
```

Generate the swarm PKI (needed for TLS/mTLS; `--host` outside localhost now requires it):
```bash
python src/gen_certs.py init-ca --out certs
```
```bash
python src/gen_certs.py node --ca-dir certs --name worker1 --ip 192.168.1.40
```

Challenge a node's Proof of Compute (loads the same layers locally as reference):
```bash
python src/attest.py --node https://192.168.1.40:8001 --layers 8-15 --tls-ca certs/ca.crt
```

Run a local tracker (reference implementation of JARVIS; makes the whole swarm runnable offline):
```bash
python src/tracker.py --port 5000 --signing-key certs/tracker.key
```

Tests — six pytest suites; the `test_phase*.py` files are standalone scripts run with `python`:
```bash
pytest tests/test_protocol.py tests/test_routing.py tests/test_tls_pow.py tests/test_client_routing.py tests/test_config_qos.py tests/test_tracker.py
```
```bash
pytest tests/test_protocol.py::test_authenticated_local_pipeline
```
```bash
python tests/test_phase1_poc.py
```
`test_phase2_gpt2.py` (downloads GPT-2), `test_phase2_5_benchmark.py` (compression benchmark, no network) and `test_phase3_e2e.py` (spawns real workers + agent, downloads Qwen, ~2 min) all hit the network or disk cache.

Build Windows executables (requires PyInstaller; deletes `build/`, `dist/` and all `*.spec` first):
```bash
python build.py
```

## Architecture

**`src/p2p_node.py` — `P2PNode`.** The transport layer, model-agnostic. An aiohttp app with `/ping`, `/forward`, `/callback`, `/attest`. `/forward` deserializes the body, runs `self.operation` (injected — the layer shard, or a lambda in tests) via `asyncio.to_thread` under a semaphore, returns **202 immediately**, and forwards the result in a background task tracked in `self._background_tasks`. Forwarding is still fire-and-forget, so a downstream error never surfaces as an HTTP status to the caller — it comes back to the client as the error envelope of `routing.build_error_envelope`. One persistent `ClientSession` is reused for every hop (a session per forward would mean a TLS handshake per hop per token).

**`src/routing.py`.** Envelope v2 = source routing: `{v, request_id, route[], hop, route_exp, route_sig, payload}`. A node forwards to `route[hop+1]`; the last entry is always the client's `/callback`, so workers no longer need `--next-node`. `authorize_route` is the SSRF boundary: a route from the network is only accepted if it's Ed25519-signed by the tracker (over the *normalized* form), or all-loopback, or the node opted into `--allow-unsigned-routes`.

**`src/tls_utils.py`.** Swarm PKI. A node's identity is its certificate, carried as SAN `dNSName = <node_id>.node.aitorrent`; clients connect to the IP but pass `server_hostname` so hostname verification validates identity independently of a dynamic IP. EC P-256 keys serve for both TLS and attestation signatures.

**`src/tracker.py` — `Tracker`.** Reference implementation of the JARVIS contract, so the swarm runs offline. `/plan` **requires** a `callback` query param and signs the route *including* it — if the client could append the last hop after signing, the final worker would deliver to any victim it chose. It only returns chains that are layer-contiguous and end at an `is_last` node. `NODE_TIMEOUT = 45s` is exactly three missed 15s heartbeats (`p2p_node.HEARTBEAT_INTERVAL`); changing one without the other silently breaks eviction.

**`src/config.py`.** Persisted JSON at `%LOCALAPPDATA%\GhostTerminal\config.json` (hotkey, tracker URL, ports, donated resources), read by the worker, the client and the installer. Every value is sanitized and clamped on load: the file is human-editable and consumed by a process with no console, so bad input must degrade to the default, never crash startup. Precedence is CLI flag > env var > config > `DEFAULTS`.

**`src/ratelimit.py`.** Token bucket keyed by cert identity or IP, with a bounded `OrderedDict` (an unbounded per-IP dict is itself a memory-exhaustion vector). Shared by node and tracker endpoints; `/attest` gets its own far stricter limiter because each call runs real inference.

**`src/pow_utils.py`.** Deterministic challenge (single-threaded, seeded), SHA-256 tensor digest, and **relative-L2 comparison under `DEFAULT_EPSILON`** — never bitwise, because `secret_sauce` downcasts to float16. `/attest` responses serialize with `use_secret_sauce=False` so the signed bytes are the verified bytes.

**`src/worker.py` — `GenericWorkerModel`.** Loads the full HF model, extracts `layers[start:end+1]`, deletes the base model. Supports two architectures, dispatched on `self.arch`: `llama/qwen` (`base_model.model.layers`, needs `position_ids` and RoPE `position_embeddings`, builds its own causal mask) and `gpt2` (`base_model.transformer.h`, positional embeddings applied client-side). Any new architecture must be added in both `GenericWorkerModel` and `ClientNodeModel`.

**`src/chat_agent.py` — `AgenticChat` + `ClientNodeModel`.** Owns the generation loop: greedy `argmax`, one full pipeline round-trip **per token** (up to 256), each correlated by a `request_id` UUID held in `self.pending_requests` as a future that `/callback` resolves. `_run_pipeline_step` wraps that with up to `MAX_ROUTE_ATTEMPTS` retries: an error envelope or timeout marks the node in `self.failed_nodes`, drops the route, and re-plans via the tracker's `/plan`. `--route`/tracker plan ⇒ envelope v2; bare `--next-node` ⇒ envelope v1 and the workers' static chaining (that fallback is what keeps the README's 2-worker setup working — a v2 route of `[nodeA, callback]` would skip worker B). `ClientNodeModel` defaults to `end_layer=7`, so workers start at layer 8. Also does agentic file writing: code blocks whose first line is a `# filename.py` comment are extracted, stashed in `self.pending_files`, and the *next* user input is consumed as the y/n answer rather than a new turn.

**`src/tensor_utils.py`.** Wire format: msgpack + zlib. Tensors become `{'_is_tensor': True, shape, dtype, data, secret_sauce}` dicts, nested inside plain dicts and lists recursively (lists exist only to carry the v2 route, hence `MAX_LIST_ITEMS = 64`, which is also `max_array_len`). "Secret sauce" = downcast float32/bfloat16 to float16 before sending (lossy, ~2x smaller). Deserialization is hardened — dtype allowlist, element/dimension/byte-length limits, bounded zlib decompression, `strict_map_key`. **Treat every limit and check here as a security boundary; do not relax them when adding features.** `MAX_COMPRESSED_BYTES` is also the aiohttp `client_max_size` on both servers.

**Security posture.** `_check_auth` accepts a request if it presented a valid swarm-CA client certificate (mTLS — OpenSSL already rejected the rest at handshake), else if `hmac.compare_digest` matches the shared `--auth-token` / `AI_TORRENT_AUTH_TOKEN`, else if there's no token and the caller is loopback. Binding outside localhost requires TLS, or an explicit `--insecure-no-tls` plus a token. `--enable-upnp` requires a token or TLS. UPnP, tracker registration (`https://jarvis.supercores.host/ai-torrent`) and the KV cache are all **off by default** — keep them opt-in. The KV cache key hashes the payload only: including `hop` or `request_id` makes it silently never hit.

**GUI/installer layer** (`ghost_terminal.py`, `setup_wizard.py`, `uninstaller.py`) is Windows-only customtkinter and wraps `AgenticChat`; `build.py` packages all four binaries, with the setup wizard embedding the other three from `dist/`. Both GUIs use `overrideredirect(True)`, which removes the OS frame — hence the hand-rolled drag and the eight resize grips in the wizard. pystray callbacks run on their own thread, so everything they touch goes through `self.after(0, ...)`; Tkinter is not thread-safe. The uninstaller cleans the registry **before** killing processes: a half-closed uninstaller may leave orphan files, but must never leave an autostart entry that resurrects the app.

## Notes

- `docs/protocol-spec.md` documents the current protocol (transport, envelopes v1/v2, route policy, Proof of Compute); the Phase 1 schema survives only as an appendix marked historical. `docs/plan-tecnico.md` holds the design rationale.
- The worker logs to `%LOCALAPPDATA%\GhostTerminal\worker.log` (rotating). This matters because `p2p_node.exe` is built `--noconsole`: without it, an argparse failure exits with code 2 and no visible trace.
- Modules import via `src.` package paths after inserting the repo root into `sys.path`; run scripts from the repo root.
- The repo has no commits yet on `master`.
