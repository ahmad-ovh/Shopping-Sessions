# Hyperion — Databricks Shopping Sessions Engine

A zero-dependency C++20 static `FROM scratch` execution engine for the **Builders Buildathon — Databricks Shopping Sessions (`shopping-sessions-5`)** track.

## Key Architectural Innovations

1. **Dual-TU Native Cascade Lake AVX-512 + AVX2 Runtime Dispatch (`src/entry_avx512.cpp`, `src/entry_avx2.cpp`, `src/main.cpp`)**:
   - Compiles the entire pipeline twice in isolated namespaces (`hyp_avx512` with `-march=cascadelake -Wa,-mbranches-within-32B-boundaries` and `hyp_avx2` with `-march=x86-64-v3 -mtune=cascadelake`) and dispatches at startup via `__builtin_cpu_supports`.
   - Unlocks 64-byte ZMM k-mask CSV scanning (`_mm512_cmpeq_epi8_mask`) and 512-bit cache-line flushes on the GCP `n2` Cascade Lake scorer while running natively on AVX2 hardware.
2. **Zero-Interning Lossless Dual-Width IDs (`src/keys.hpp`)**:
   - **Domain 0 (`DOM_SHORT`, 87.5% of v5 IDs)**: `prefix + 1..8 ASCII` packed verbatim into a 64-bit word (`bswap64` preserves exact byte-wise ordering).
   - **Domain 1 (`DOM_HEX32`, 12.5% of v5 IDs)**: `prefix + 32 lowercase hex` packed via 10-instruction AVX2 SIMD (`parse_hex32_avx2`) into a lossless 128-bit `Hex128` (`(hi, lo)` big-endian preserves exact byte-wise ordering; 5-instruction AVX2 decode).
   - **Domain 2 (`DOM_EXOTIC`, `< 0.001%` + arbitrary fixtures)**: Exact mutex-guarded byte arena with O(1) cross-domain comparison (`hex128_top8_be`).
   - Eliminates global string-interning hash tables and mutex contention on 99.9995% of v5 rows.
3. **Fused 1-Pass Parallel Lock-Free DSU + 1 MiB Register-Blocked Bloom + 16-Byte Compact Alias Table (`src/catalog_links.hpp`)**:
   - Parses `links/*.csv`, interns nodes, and unites components in a single parallel lock-free pass (overlapped with zero-allocation `products.csv` and `campaigns.csv` loading).
   - Counts exact component roots in Pass 2 to size the non-canonical `ShortAliasSlot` (16 B) and `HexAliasSlot` tables tightly, populates a 1 MiB L2/L3-resident 1-word register-blocked Bloom filter in Pass 3, and immediately releases all temporary DSU pages via parallel `MADV_DONTNEED`.
4. **Zero-Overlap 3-Mask Streaming SIMD CSV Scanner + Branchless 79%-Quote Stripping + 2-Month Prefix SWAR Timestamps (`src/csv_time.hpp`)**:
   - Classifies each 64-byte aligned block once into separator, newline (`nlm`), and quote bitmasks.
   - Strips quotes branchlessly via `base[p - isq]` (zero branch mispredictions on the 79% of event lines that carry quoted fields) and validates line quote parity via a single `qbefore` popcount per line.
   - Precomputes 8-byte `"YYYY-MM-"` ASCII prefixes for the active window months (`a == ym0_ascii`) plus zero-modulo overlapping SWAR for 13-digit `F3` epoch milliseconds.
5. **Single-Scatter In-Place Deduplication + L1 Write-Combining Staging (`src/pipeline.hpp`)**:
   - Valid events scatter **once** directly into person partitions (`ShortPGroup5`: 5 events packed SoA into one 64-byte cache line; 3% purchases separated into `PurchBlock`), while compact 16-byte `ShortPStageMeta`/`ShortEStageMeta`/`ShortClickStageMeta` avoid touching 4 KiB block headers on 75–80% of events.
   - Disjoint L2-resident dedup partitions (`16B ShortERec`, `24B LongERec`) tombstone only the ~3% losing duplicate copies in place via `__atomic_store_n(..., DEAD_TS, __ATOMIC_RELAXED)`.
   - `SlabArena::rewind()` recycles `e_arena`'s already-faulted 2 MiB THP huge pages after `dedup_phase` to back `UserDailyCols` in Phase 3 with zero new page faults.
6. **Zero Global Click Dedup + Lazy `click_id` Tie-Breaking + Sort-Free 30-Minute Bitmask Sessionization (`src/pipeline.hpp`)**:
   - Duplicate clicks carry identical `(user_id, campaign_id, ts)` and cannot alter latest-click credit or distinct-campaign 7-day assists; `click_id` is read lazily from the `mmap`ped click file only on same-millisecond multi-campaign ties (`< 0.5%` of clicks).
   - LPT (Longest-Processing-Time) scheduling processes whale partitions first; single-event (`ne == 1`) and small (`ne <= 8`) users use register/stack fast paths, while large users and 1.2M-event whales collapse into 30-minute bins via an 11-word `bmap` bitmask in $O(N)$ time with zero sorting.
   - Campaign-day counters are split into L2-resident 8-byte `CampDayHot` (`224 KiB`/thread) and cold 16-byte `CampDayPurch` (touched on only 3% of sessions).
7. **Zero-Copy `writev()` Custom RLE_DICTIONARY Parquet Writer + Parallel 64 MiB-Chunked `MADV_DONTNEED` (`src/parquet.hpp`)**:
   - Emits `user_id` via `RLE_DICTIONARY` (1 dictionary entry + 1 RLE run per person), `top_category` via 21-entry dictionary + SIMD `_mm_movemask_epi8` 1-bit definition levels, and adaptive `INT32`/`INT64` `revenue_cents` via scatter-gather `writev()`.
   - Before calling `_exit(0)`, all threads release both file-backed `MAP_PRIVATE` PTEs and anonymous huge-page arenas in 64 MiB chunks via `MADV_DONTNEED`, eliminating single-threaded kernel `exit_mmap()` teardown.

## Build & Run

```powershell
docker build -t hyperion:latest .
python scripts/test_and_bench.py fixtures hyperion:latest
python scripts/test_and_bench.py dev hyperion:latest
```

## Documentation

- [Competitive Reconnaissance & Strategy Analysis](docs/RESEARCH.md)
- [Experiments, Progression & Ablations Log](docs/EXPERIMENTS.md)
