# Hyperion Experiments Log

All local benchmarks run on Intel Core Ultra 7 258V (8 logical CPUs, AVX2) inside Docker Desktop (WSL2, `--cpus 8 --memory 14g`, native ext4 Docker volume `ss-data:/vol:ro`, tmpfs `/output`, exact scorer flags).

## Calibrated Competitor Baselines (`p20`: 20M events, 5M clicks, 2.0 GB CSV)

| Engine | Language | Local `p20` Warm Median | Local `p20` Min | Notes |
| --- | --- | --- | --- | --- |
| `comp-mego74` | C++17 | **2.4898 s** | **2.4178 s** | Single-scatter in-place dedup, global atomic `IDS`, PLAIN Parquet |
| `comp-aqilc` | C | 3.5286 s | 3.1319 s | Two-scatter dedup |
| `comp-dulaj` | C++17 | 6.9228 s | 6.3994 s | Two-scatter dedup |

## Hyperion Progression

| Version | Fixtures | 1M Dev | Local `p20` Warm Median | Local `p20` Min | Key Changes & Profile Breakdown (ms) |
| --- | --- | --- | --- | --- | --- |
| **v1** | **36/36** | **MATCH** | 3.8720 s | 3.6938 s | Initial clean C++20 engine: zero-interning 64b+128b AVX2 hex keys, single-scatter in-place dedup, sort-free 30-min bitmask sessions, RLE_DICTIONARY Parquet. Profile: `links 754 \| parse 1912 \| dedup 71 \| sess+pq 470 \| camp 9 \| madv 36` |
