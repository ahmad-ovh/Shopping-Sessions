#!/usr/bin/env python3
"""
Windows-first test & benchmark harness for Hyperion (Shopping Sessions).
Runs all 36 official fixtures, verifies exact match on 1M dev vs dev-answer,
and benchmarks 20M practice dataset (/vol/p20) inside Docker with scorer flags.
"""
import os
import sys
import time
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(r"c:\Users\User\Documents\Buildathon\databricks")
KIT  = ROOT / ".local" / "kit"
FX_DIR = KIT / "starter-duckdb" / "starter-duckdb" / "fixtures"
DEV_ANSWER = KIT / "dev-answer" / "dev-answer"
SCRIPTS_DIR = KIT / "starter-duckdb" / "starter-duckdb" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
import polars as pl
from run_fixtures import SCHEMAS, load_actual, load_expected, rows as sorted_rows


def docker_run_bind(image: str, data_dir: Path, out_dir: Path, cpus="8", mem="14g", extra_env=None):
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--tmpfs", "/scratch:rw,size=512m,mode=1777",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "4096", "--cpus", cpus, "--memory", mem, "--memory-swap", mem,
        "--user", "1000:1000",
        "-v", f"{str(data_dir)}:/data:ro",
        "-v", f"{str(out_dir)}:/output",
        "-e", "DATA_DIR=/data", "-e", "OUTPUT_DIR=/output",
        "-e", "SCRATCH_DIR=/scratch", "-e", "HOME=/scratch",
        "-e", f"NCPU={cpus}", "-e", f"MEM_GB={mem[:-1]}",
    ]
    if extra_env:
        for k, v in extra_env.items():
            cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    for attempt in range(3):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if p.returncode != 125:
            break
        time.sleep(2)
    return p, dt


def run_fixtures(image: str):
    cases = sorted(d for d in os.listdir(FX_DIR) if (FX_DIR / d).is_dir())
    failed = []
    # Run all 36 fixtures in a single container? No, run each in a scratch container or batch!
    # Running 36 separate docker containers on Windows takes ~20s; batch via a single mount is also nice,
    # but let's run exact per-fixture Docker invocation for faithful isolation (parallel 4x -> ~5s!).
    from concurrent.futures import ThreadPoolExecutor

    def check_one(case):
        cdir = FX_DIR / case
        data = cdir / "data"
        if not data.is_dir():
            data = cdir / "input"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = Path(tmp)
            p, _ = docker_run_bind(image, data, out, cpus="2", mem="1g")
            if p.returncode != 0:
                return (case, False, f"exit {p.returncode}: {p.stderr[-400:]}")
            for table, schema in SCHEMAS.items():
                exp = sorted_rows(load_expected(cdir / "expected" / f"{table}.csv", schema))
                try:
                    act = sorted_rows(load_actual(out / table, schema))
                except Exception as e:
                    return (case, False, f"{table} read error: {e}")
                if exp != act:
                    return (case, False, f"{table} MISMATCH\n  exp={exp[:5]}\n  act={act[:5]}")
            return (case, True, "ok")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for case, ok, msg in ex.map(check_one, cases):
            if ok:
                print(f"  ok   {case}")
            else:
                failed.append(case)
                print(f"  FAIL {case}: {msg}")
    print(f"Fixtures: {len(cases) - len(failed)}/{len(cases)} passed", flush=True)
    return len(failed) == 0


def check_dev(image: str):
    print("\n=== Checking 1M dev (seed 1) vs official dev-answer ===", flush=True)
    out = ROOT / ".local" / "out_dev"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--tmpfs", "/scratch:rw,size=512m,mode=1777",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "4096", "--cpus", "8", "--memory", "14g", "--memory-swap", "14g",
        "--user", "1000:1000",
        "-v", "ss-data:/vol:ro",
        "-v", f"{str(out)}:/output",
        "-e", "DATA_DIR=/vol/dev", "-e", "OUTPUT_DIR=/output",
        "-e", "SCRATCH_DIR=/scratch", "-e", "HOME=/scratch",
        "-e", "NCPU=8", "-e", "MEM_GB=14",
        "-e", "HYPERION_PROFILE=1",
        image,
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.stderr:
        print(p.stderr.strip())
    if p.returncode != 0:
        print(f"DEV RUN FAILED rc={p.returncode}")
        return False
    print(f"dev container wall: {dt:.4f}s")
    matched = True
    for table, schema in SCHEMAS.items():
        a = load_actual(out / table, schema)
        b = load_expected(DEV_ANSWER / f"{table}.csv", schema)
        keys = [c for c in ("user_id", "campaign_id", "day") if c in schema]
        dup_a = a.height - a.n_unique(subset=keys)
        dup_b = b.height - b.n_unique(subset=keys)
        only_a = a.join(b, on=list(schema), how="anti", nulls_equal=True)
        only_b = b.join(a, on=list(schema), how="anti", nulls_equal=True)
        print(f"  {table}: rows {a.height} vs {b.height}; only_A={only_a.height}; only_B={only_b.height}; dups={dup_a}/{dup_b}")
        if dup_a or dup_b or only_a.height or only_b.height:
            matched = False
            print("  only_A:", only_a.sort(keys).head(5))
            print("  only_B:", only_b.sort(keys).head(5))
    print("DEV RESULT:", "MATCH" if matched else "DIFFER", flush=True)
    return matched


def bench_p20(image: str, runs=5, profile=True):
    print(f"\n=== Benchmarking {image} on /vol/p20 ({runs} runs) ===", flush=True)
    times = []
    for i in range(runs):
        cmd = [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--tmpfs", "/output:rw,size=512m,mode=1777",
            "--tmpfs", "/scratch:rw,size=512m,mode=1777",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "4096", "--cpus", "8", "--memory", "14g", "--memory-swap", "14g",
            "--user", "1000:1000",
            "-v", "ss-data:/vol:ro",
            "-e", "DATA_DIR=/vol/p20", "-e", "OUTPUT_DIR=/output",
            "-e", "SCRATCH_DIR=/scratch", "-e", "HOME=/scratch",
            "-e", "NCPU=8", "-e", "MEM_GB=14",
        ]
        if profile and i == runs - 1:
            cmd += ["-e", "HYPERION_PROFILE=1"]
        cmd.append(image)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if p.returncode != 0:
            print(f"  run {i} FAILED rc={p.returncode}: {p.stderr[-500:]}")
            return None
        times.append(dt)
        print(f"  p20 run {i}: {dt:.4f}s", flush=True)
        if profile and i == runs - 1 and p.stderr:
            print(p.stderr.strip())
    warm = times[1:] if len(times) > 1 else times
    med = statistics.median(warm)
    print(f"  >>> {image} p20 warm median={med:.4f}s (min={min(warm):.4f}s)", flush=True)
    return med


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    img = sys.argv[2] if len(sys.argv) > 2 else "hyperion:v1"
    if mode in ("all", "fixtures"):
        if not run_fixtures(img):
            sys.exit(1)
    if mode in ("all", "dev"):
        if not check_dev(img):
            sys.exit(1)
    if mode in ("all", "bench"):
        bench_p20(img, runs=5, profile=True)
