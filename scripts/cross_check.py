#!/usr/bin/env python3
"""
Cross-checks Hyperion against an independent oracle (comp-mego74) across
multiple random seeds and scales (including 20M p20!).
"""
import sys, shutil, subprocess
from pathlib import Path

ROOT = Path(r"c:\Users\User\Documents\Buildathon\databricks")
SCRIPTS_DIR = ROOT / ".local" / "kit" / "starter-duckdb" / "starter-duckdb" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from run_fixtures import SCHEMAS, load_actual

def run_img_to_dir(img, vol_sub, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--tmpfs", "/scratch:rw,size=512m,mode=1777",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "4096", "--cpus", "8", "--memory", "14g", "--memory-swap", "14g",
        "--user", "1000:1000",
        "-v", "ss-data:/vol:ro",
        "-v", f"{str(out_dir)}:/output",
        "-e", f"DATA_DIR=/vol/{vol_sub}", "-e", "OUTPUT_DIR=/output",
        "-e", "SCRATCH_DIR=/scratch", "-e", "HOME=/scratch",
        "-e", "NCPU=8", "-e", "MEM_GB=14",
        img,
    ]
    for attempt in range(3):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 125:
            break
        import time; time.sleep(2)
    if p.returncode != 0:
        raise RuntimeError(f"{img} failed rc={p.returncode}: {p.stderr[-500:]}")

def compare_dirs(dir_a: Path, dir_b: Path):
    ok = True
    for table, schema in SCHEMAS.items():
        a = load_actual(dir_a / table, schema)
        b = load_actual(dir_b / table, schema)
        keys = [c for c in ("user_id", "campaign_id", "day") if c in schema]
        dup_a = a.height - a.n_unique(subset=keys)
        dup_b = b.height - b.n_unique(subset=keys)
        only_a = a.join(b, on=list(schema), how="anti", nulls_equal=True)
        only_b = b.join(a, on=list(schema), how="anti", nulls_equal=True)
        print(f"  {table}: rows {a.height} vs {b.height}; only_A={only_a.height}; only_B={only_b.height}; dups={dup_a}/{dup_b}")
        if dup_a or dup_b or only_a.height or only_b.height:
            ok = False
            print("  only_A:", only_a.sort(keys).head(5))
            print("  only_B:", only_b.sort(keys).head(5))
    return ok

if __name__ == "__main__":
    img_a = sys.argv[1] if len(sys.argv) > 1 else "hyperion:v2"
    img_b = sys.argv[2] if len(sys.argv) > 2 else "comp-mego74"
    vol   = sys.argv[3] if len(sys.argv) > 3 else "p20"
    out_a = ROOT / ".local" / "xchk_a"
    out_b = ROOT / ".local" / "xchk_b"
    print(f"=== Cross-checking {img_a} vs {img_b} on /vol/{vol} ===", flush=True)
    run_img_to_dir(img_a, vol, out_a)
    run_img_to_dir(img_b, vol, out_b)
    ok = compare_dirs(out_a, out_b)
    print("CROSS-CHECK:", "EXACT MATCH" if ok else "DIFFER")
    sys.exit(0 if ok else 1)
