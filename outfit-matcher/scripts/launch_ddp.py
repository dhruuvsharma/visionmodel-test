"""Minimal multi-GPU DDP launcher for Windows (avoids torchrun's libuv-dependent
elastic rendezvous, which is broken on some torch Windows builds).

Picks a free port, spawns one subprocess per GPU with the standard
MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK / LOCAL_RANK env vars, and
streams their output. Each worker calls init_process_group("env://") in
outfit_matcher.engine (which respects USE_LIBUV=0).

Usage:
    python scripts/launch_ddp.py --nproc 2 --config configs/shirts.yaml [--data-override D:/data/shirts]
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-override", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="extra args passed to train module")
    args = ap.parse_args()

    port = free_port()
    base_env = {**os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "WORLD_SIZE": str(args.nproc),
                "USE_LIBUV": "0",          # TCPStore without libuv (Windows-safe)
                "PYTHONPATH": str(ROOT),
                "OMP_NUM_THREADS": "4"}

    procs = []
    for rank in range(args.nproc):
        env = {**base_env, "RANK": str(rank), "LOCAL_RANK": str(rank)}
        cmd = [sys.executable, "-m", "outfit_matcher.train", "--config", args.config]
        if args.data_override:
            cmd += ["--data-override", args.data_override]
        cmd += args.extra
        print(f"[launcher] spawn rank {rank}: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(ROOT)))

    rc = 0
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(1)
        for p in procs:
            rc = max(rc, p.wait())
    except KeyboardInterrupt:
        print("[launcher] Ctrl+C -> terminating workers", flush=True)
        for p in procs:
            p.terminate()
        rc = 130
    print(f"[launcher] done, rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
