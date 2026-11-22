#!/usr/bin/env python3
"""
Interleaved ABBA benchmark on /vol/p20 to eliminate laptop thermal drift.
Prints both wall time and internal process time (when profile is enabled).
Usage: python scripts/bench_ab.py IMAGE_A IMAGE_B [ROUNDS=3]
"""
import sys, time, statistics, subprocess, re

def run_one(img, profile=False):
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
    if profile:
        if "mego74" in img:
            cmd += ["-e", "PIPE_VERBOSE=1"]
        else:
            cmd += ["-e", "HYPERION_PROFILE=1"]
    cmd.append(img)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{img} failed rc={p.returncode}: {p.stderr[-400:]}")
    internal = None
    if profile:
        # Parse last total ms
        ms_matches = re.findall(r"total\s+([\d.]+)\s*ms", p.stderr)
        if ms_matches:
            internal = float(ms_matches[-1]) / 1000.0
        else:
            m2 = re.findall(r"\[\s*([\d.]+)\s*ms\]\s*released", p.stderr)
            if m2:
                internal = float(m2[-1]) / 1000.0
    return dt, internal, p.stderr.strip()

if __name__ == "__main__":
    img_a = sys.argv[1]
    img_b = sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    # Warmup
    run_one(img_a, profile=False)
    run_one(img_b, profile=False)
    wa, wb, ia, ib = [], [], [], []
    last_err_a = last_err_b = ""
    for r in range(rounds):
        order = [(img_a, wa, ia), (img_b, wb, ib)] if (r % 2 == 0) else [(img_b, wb, ib), (img_a, wa, ia)]
        for img, wl, il in order:
            w, it, err = run_one(img, profile=True)
            wl.append(w)
            if it is not None:
                il.append(it)
            if img == img_a: last_err_a = err
            else: last_err_b = err
            print(f"  r{r} {img:16s}: wall={w:.4f}s internal={it:.4f}s" if it else f"  r{r} {img:16s}: wall={w:.4f}s", flush=True)
    print(f"\n=== SUMMARY ({rounds} interleaved rounds) ===")
    print(f"  {img_a:16s}: wall med={statistics.median(wa):.4f}s (min={min(wa):.4f}s) | internal med={statistics.median(ia):.4f}s (min={min(ia):.4f}s)")
    print(f"  {img_b:16s}: wall med={statistics.median(wb):.4f}s (min={min(wb):.4f}s) | internal med={statistics.median(ib):.4f}s (min={min(ib):.4f}s)")
    print(f"\n--- {img_a} last profile ---\n{last_err_a}")
