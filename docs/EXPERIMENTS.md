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
| **v4** | **36/36** | **MATCH** | 2.5407 s (1.9256 s int) | 2.4858 s (1.7189 s int) | Zero-alloc `products`/`campaigns`, fused 1-pass lock-free DSU + 1 MB Bloom + 16 MB `short_tab`, 1-`qbefore`/line `StreamScanner`, compact `PendEvent`. `mmap` beat `pread` (-171 ms); `P=512` beat `P=1024` (-161 ms). Exact 20M cross-check MATCH vs `comp-mego74`. |
| **v5** | **36/36** | **MATCH** | — | 1.7357 s int (`NT=0`) | Dual-TU native Cascade Lake AVX-512 + AVX2 runtime dispatch, 2-month `"YYYY-MM-"` fast-path + zero-mod F3, zero-block-touch `ShortP/ShortE/ShortClick` 16B stage metas, `HugeBuf` Parquet + SIMD `def_levels` + zero-copy `writev()` (`user_pq` 10–20 ms!). **Ablation**: Non-temporal `_mm256_stream_si256` stores regressed `parse+scatter` by +1,457 ms (WC buffer thrash across 2,048 streams) → rejected. `P=512, DS=1024` decisively beat `P=256` and `DS=256/512`. |
| **v5.2** | **36/36** | **MATCH** | **2.3443 s** (**1.6875 s** int) | **2.0847 s** (**1.6538 s** int) | Restored branchless quote stripping (`base[p-isq]` — 79% of event lines are quoted!) + `nlm` bitmask; shrunk `s_nodes` 64→32 MB & `short_tab` 32→16 MB via exact root count; lock-free `canon_overflow` in `links:pass3` (`cat+links+bloom` 236→146 ms); single-branch `hit_idx` + split L1 `P`/`E` scatter; LPT whale-first scheduling + flat `Session` array + prefetched `dedup`/`ShortP`/`ShortClick`; **parallel 64 MiB-chunked `MADV_DONTNEED` of 2.0 GB file mappings + anonymous arenas (-411 ms wall!)**. Exact 20M cross-check MATCH vs `comp-mego74`. |
