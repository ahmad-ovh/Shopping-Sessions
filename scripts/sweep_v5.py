#!/usr/bin/env python3
"""Interleaved A/B sweep for Hyperion configs vs comp-mego74 on /vol/p20."""
import re
import sys
import time
import statistics
import subprocess

def run_one(image, env_dict, data_path="/vol/p5", mem="6g"):
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--tmpfs", "/scratch:rw,size=512m,mode=1777",
        "--mount", "type=tmpfs,destination=/output",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "4096", "--cpus", "8", "--memory", mem, "--memory-swap", mem,
        "--user", "1000:1000",
        "-v", "ss-data:/vol:ro",
        "-e", f"DATA_DIR={data_path}", "-e", "OUTPUT_DIR=/output",
        "-e", "SCRATCH_DIR=/scratch", "-e", "HOME=/scratch",
        "-e", "NCPU=8", "-e", "MEM_GB=14",
    ]
    for k, v in env_dict.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    for attempt in range(4):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if p.returncode == 0:
            break
        if attempt < 3 and (p.returncode in (1, 2, 125) and "hyperion fatal" not in p.stderr):
            time.sleep(2.0)
            continue
        raise RuntimeError(f"{image} failed rc={p.returncode}: {p.stderr[-400:]}")
    internal = None
    m = re.findall(r"total\s+([\d.]+)\s+ms", p.stderr)
    if m:
        internal = float(m[-1]) / 1000.0
    m_se = re.search(r"total=([\d.]+)ms", p.stderr)
    if internal is None and m_se:
        internal = float(m_se.group(1)) / 1000.0
    m2 = re.search(r"TOTAL:\s+([\d.]+)s", p.stdout + "\n" + p.stderr)
    if internal is None and m2:
        internal = float(m2.group(1))
    return dt, internal, p.stderr.strip()

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "part"
    if mode == "part":
        configs = [
            ("v5-P256-DS256",  "hyperion:v5", {"HYPERION_PROFILE": "1", "HYPERION_NT": "0", "HYPERION_P": "256", "HYPERION_DS": "256"}),
            ("v5-P256-DS512",  "hyperion:v5", {"HYPERION_PROFILE": "1", "HYPERION_NT": "0", "HYPERION_P": "256", "HYPERION_DS": "512"}),
            ("v5-P512-DS512",  "hyperion:v5", {"HYPERION_PROFILE": "1", "HYPERION_NT": "0", "HYPERION_P": "512", "HYPERION_DS": "512"}),
            ("v5-P512-DS1024", "hyperion:v5", {"HYPERION_PROFILE": "1", "HYPERION_NT": "0", "HYPERION_P": "512", "HYPERION_DS": "1024"}),
        ]
        rounds = 2
    else:
        img_new = sys.argv[2] if len(sys.argv) > 2 else "hyperion:latest"
        img_base = sys.argv[3] if len(sys.argv) > 3 else "comp-aryan"
        data_path = sys.argv[4] if len(sys.argv) > 4 else "/vol/p5"
        configs = [
            (img_base,      img_base,      {"HYPERION_PROFILE": "1", "SE_TIMING": "1", "SS_VERBOSE": "1"}),
            (img_new,       img_new,       {"HYPERION_PROFILE": "1", "SE_TIMING": "1", "SS_VERBOSE": "1"}),
            ("comp-mego74", "comp-mego74", {}),
        ]
        rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    # Warmup
    run_one(configs[0][1], configs[0][2], data_path=data_path)

    res = {name: {"wall": [], "int": [], "last": ""} for name, _, _ in configs}
    for r in range(rounds):
        order = configs if (r % 2 == 0) else list(reversed(configs))
        for name, img, env in order:
            w, i, err = run_one(img, env, data_path=data_path)
            res[name]["wall"].append(w)
            if i is not None:
                res[name]["int"].append(i)
            res[name]["last"] = err
            istr = f"{i:.4f}s" if i is not None else "n/a"
            print(f"  round {r+1} {name:16s}: wall={w:.4f}s  internal={istr}", flush=True)

    print("\n=== SUMMARY ===")
    for name, _, _ in configs:
        ws = res[name]["wall"]
        ins = res[name]["int"]
        im = f"{statistics.median(ins):.4f}s (min {min(ins):.4f}s)" if ins else "n/a"
        print(f"  {name:16s}: wall med={statistics.median(ws):.4f}s (min {min(ws):.4f}s) | internal med={im}")
    print("\n=== LAST PROFILES ===")
    for name, _, _ in configs:
        if res[name]["last"]:
            print(f"--- {name} ---")
            print(res[name]["last"])

if __name__ == "__main__":
    main()
