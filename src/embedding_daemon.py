#!/usr/bin/env python3
"""Embedding Daemon — text-embedding-3-large via OpenAI API over Unix socket.

v2: Switched from local fastembed (jina-v2-base-zh, 512d) to API
    (text-embedding-3-large, 3072d) for dimension consistency across
    the full system. Trade-off: ~200ms vs ~25ms per query.

Usage:
    python3 scripts/embedding-daemon.py start &
    echo '{"texts":["project status","team portfolio"]}' | nc -U /tmp/embedding-daemon.sock
    python3 scripts/embedding-daemon.py stop

Protocol: JSON over Unix domain socket
    Request:  {"texts": ["text1", "text2", ...]}
    Response: {"embeddings": [[0.1, 0.2, ...], [0.3, ...]], "dim": 3072, "ms": 200}
"""

import json
import os
import signal
import socket
import sys
import time

SOCKET_PATH = "/tmp/embedding-daemon.sock"
PID_FILE = "/tmp/embedding-daemon.pid"
MODEL_NAME = "text-embedding-3-large"
MAX_BATCH = 64
BUFFER_SIZE = 1024 * 1024  # 1MB


def _get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def start():
    """Start the embedding daemon."""
    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.kill(pid, 0)
            print(f"Daemon already running (PID {pid})", file=sys.stderr)
            sys.exit(1)
        except (OSError, ValueError):
            pass

    print(f"Initializing API client for {MODEL_NAME}...", file=sys.stderr)
    client = _get_client()

    # Warm up — verify API connectivity
    t0 = time.time()
    try:
        resp = client.embeddings.create(model=MODEL_NAME, input=["warmup"])
        dim = len(resp.data[0].embedding)
        print(f"API ready in {(time.time()-t0)*1000:.0f}ms, dim={dim}", file=sys.stderr)
    except Exception as e:
        print(f"API warmup failed: {e}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    server.settimeout(1.0)

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    print(f"Daemon listening on {SOCKET_PATH} (PID {os.getpid()})", file=sys.stderr)

    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    request_count = 0

    while running:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            data = b""
            while True:
                chunk = conn.recv(BUFFER_SIZE)
                if not chunk:
                    break
                data += chunk
                try:
                    json.loads(data)
                    break
                except json.JSONDecodeError:
                    continue

            if not data:
                conn.close()
                continue

            req = json.loads(data)
            texts = req.get("texts", [])

            if not texts or len(texts) > MAX_BATCH:
                resp = {"error": f"texts must be 1-{MAX_BATCH} items"}
            else:
                t_start = time.time()
                # Truncate long texts for API
                truncated = [t[:8000] for t in texts]
                api_resp = client.embeddings.create(model=MODEL_NAME, input=truncated)
                embs = [item.embedding for item in api_resp.data]
                elapsed_ms = (time.time() - t_start) * 1000
                request_count += 1
                resp = {
                    "embeddings": embs,
                    "dim": len(embs[0]) if embs else 0,
                    "ms": round(elapsed_ms, 1),
                    "n": request_count,
                }

            conn.sendall(json.dumps(resp).encode())
        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode())
            except:
                pass
        finally:
            conn.close()

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    if os.path.exists(PID_FILE):
        os.unlink(PID_FILE)
    print("Daemon stopped.", file=sys.stderr)


def stop():
    """Stop the daemon."""
    if not os.path.exists(PID_FILE):
        print("No daemon running.", file=sys.stderr)
        return
    try:
        pid = int(open(PID_FILE).read().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}", file=sys.stderr)
    except (OSError, ValueError) as e:
        print(f"Error stopping daemon: {e}", file=sys.stderr)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        start()
    elif cmd == "stop":
        stop()
    else:
        print(f"Usage: {sys.argv[0]} [start|stop]", file=sys.stderr)
