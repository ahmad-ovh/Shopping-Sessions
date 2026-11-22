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
| **v2** | **36/36** | **MATCH** | 4.0449 s | 3.7201 s | Parallel lock-free DSU + 16B `ShortAliasSlot` cut `links` 754→250 ms; tile `any_quote` loop regressed `parse` 1912→2228 ms. THP check proved `MADV_HUGEPAGE` saves ~1.0 s (`3.63 s` vs `4.56 s`). |
| **v3** | **36/36** | **MATCH** | 2.6451 s | 2.5881 s | Zero-overlap 2-mask streaming SIMD `Scanner` + branchless `qbefore` + zero-hash SWAR `dt19`/`ingest_secs` + single-mmap `SlabArena` + `ShortPGroup5` 64B cache-line write-combining + single-hash `sessions` + zero-copy `rev32` Parquet (**-1.23 s vs v1!**) |
| **v4** | **36/36** | **MATCH** | **2.5407 s** (1.9256 s int) | **2.4858 s** (1.7189 s int) | Zero-alloc `products`/`campaigns`, fused 1-pass lock-free DSU + 1 MB Bloom + 16 MB `short_tab`, 1-`qbefore`/line `StreamScanner`, compact `PendEvent`. **Beats `comp-mego74` in interleaved ABBA on `p20` by -314 ms internal (-14%, 1.9256 s vs 2.2393 s) and -1.58 s wall (2.5407 s vs 4.1214 s)!** `mmap` beat `pread` by -171 ms internal (`1.9256 s` vs `2.0967 s`); `P=512` beat `P=1024` by -161 ms (`1.9256 s` vs `2.0868 s`). Exact 20M cross-check MATCH vs `comp-mego74` (2,250,361 `user_daily` + 28,014 `campaign_daily`). |
