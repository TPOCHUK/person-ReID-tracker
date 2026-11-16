"""Check a real local server with video input and concurrent HTTP requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen, Request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="videos/test.mp4")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--clients", type=int, default=8)
    args = parser.parse_args()
    output = ROOT / "benchmark_results" / "review"
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def request(path, method="GET"):
        start = time.perf_counter()
        with urlopen(Request(base + path, method=method), timeout=30) as response:
            body = response.read()
            assert response.status == 200
        return json.loads(body), (time.perf_counter() - start) * 1000

    env = os.environ.copy()
    env["CAMERA_SOURCE"] = str((ROOT / args.source).resolve())
    result = {"source": Path(args.source).name, "clients": args.clients}
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                try:
                    request("/status")
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Server failed to start; see server.log")
                    time.sleep(0.25)
            result["start"] = request("/system/start", "POST")[0]
            deadline = time.monotonic() + 60
            while True:
                target, _ = request("/target")
                if target["total_ms"] > 0:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("No processed frame")
                time.sleep(0.1)
            result["target"] = target
            with urlopen(base + "/video/stream", timeout=30) as stream:
                chunk = stream.read(8192)
                assert b"Content-Type: image/jpeg" in chunk and b"\xff\xd8" in chunk
                result["mjpeg"] = "JPEG received"
            with ThreadPoolExecutor(max_workers=args.clients) as pool:
                results = list(
                    pool.map(
                        lambda i: request("/target" if i % 2 else "/status"), range(args.requests)
                    )
                )
            latency = [elapsed for _, elapsed in results]
            result["http"] = {
                "successful": len(results),
                "p50_ms": round(float(np.percentile(latency, 50)), 2),
                "p95_ms": round(float(np.percentile(latency, 95)), 2),
                "max_ms": round(max(latency), 2),
            }
            result["stop"] = request("/system/stop", "POST")[0]
            stopped, _ = request("/target")
            assert stopped["state"] == "idle" and not stopped["found"]
            assert not request("/status")[0]["running"]
            request("/system/start", "POST")
            assert request("/status")[0]["running"]
            request("/system/stop", "POST")
            result["restart"] = "passed"
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    path = output / "server_check.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
