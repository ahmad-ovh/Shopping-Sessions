#!/usr/bin/env python3
"""Controlled A/B sweep of Hyperion v4 configs vs comp-mego74 on /vol/p20."""
import subprocess, time, statistics, re

configs = [
    ("v4-mmap-512",   "hyperion:v4", {"HYPERION_PROFILE": "1", "HYPERION_IO": "mmap",  "HYPERION_P": "512",  "HYPERION_DS": "1024"}),
    ("v4-pread-512",  "hyperion:v4", {"HYPERION_PROFILE": "1", "HYPERION_IO": "pread", "HYPERION_P": "512",  "HYPERION_DS": "1024"}),
    ("v4-mmap-1024",  "hyperion:v4", {"HYPERION_PROFILE": "1", "HYPERION_IO": "mmap",  "HYPERION_P": "1024", "HYPERION_DS": "2048"}),
    ("v4-pread-1024", "hyperion:v4", {"HYPERION_PROFILE": "1", "HYPERION_IO": "pread", "HYPERION_P": "1024", "HYPERION_DS": "2048"}),
    ("comp-mego74",   "comp-mego74", {"PIPE_VERBOSE": "1"}),
]

def run_one(img, env):
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
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(img)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{img} failed rc={p.returncode}: {p.stderr[-400:]}")
    ms = re.findall(r"total\s+([\d.]+)\s*ms", p.stderr)
    if ms:
        internal = float(ms[-1]) / 1000.0
    else:
        m2 = re.findall(r"\[\s*([\d.]+)\s*ms\]\s*released", p.stderr)
        internal = float(m2[-1]) / 1000.0 if m2 else dt
    return dt, internal, p.stderr.strip()

# Warmup
run_one("hyperion:v4", {"HYPERION_IO": "mmap"})

res = {name: ([], [], "") for name, _, _ in configs}
for r in range(3):
    order = configs if r % 2 == 0 else list(reversed(configs))
    for name, img, env in order:
        w, it, err = run_one(img, env)
        res[name][0].append(w)
        res[name][1].append(it)
        res[name] = (res[name][0], res[name][1], err)
        print(f"  r{r} {name:15s}: wall={w:.4f}s  internal={it:.4f}s", flush=True)

print("\n=== A/B SWEEP SUMMARY (3 interleaved rounds on /vol/p20) ===")
for name, _, _ in configs:
    ws, its, _ = res[name]
    print(f"  {name:15s}: wall med={statistics.median(ws):.4f}s (min={min(ws):.4f}s) | internal med={statistics.median(its):.4f}s (min={min(its):.4f}s)")

best_name = min((n for n, _, _ in configs if n != "comp-mego74"), key=lambda n: statistics.median(res[n][1]))
print(f"\n--- Best Hyperion ({best_name}) profile ---\n{res[best_name][2]}")
