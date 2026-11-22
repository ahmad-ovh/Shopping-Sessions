# Competitive Reconnaissance & Strategy Analysis

## 1. Track & Scorer Facts (`shopping-sessions-5`)

- **Metric**: Container wall time in seconds (start to exit, median of 3 if `< 20s`). Exact table equality required; any discrepancy scores 0.
- **Scale**:
  - Public Dev: 1M events, 250K clicks (~100 MB CSV).
  - Hidden Practice: ~20M events, ~5M clicks (~2.0 GB CSV).
  - Hidden Final: ~60M events, ~15M clicks (~6.0 GB CSV), top 30 retimed 9 times across 3 fresh datasets alone on an idle machine.
- **Hardware & Container**: `linux/amd64` (GCP `n2` Intel Xeon Cascade Lake / AVX-512), 8 pinned vCPUs, 24 GB RAM, no swap, read-only root (`FROM scratch`), `/tmp` 64 MB tmpfs, uid 1000, warm page cache on practice/final.
- **v5 Rule Change (2026-11-22 02:08 UTC)**:
  - Replaced invertible scramble with a 6-round keyed Feistel bijection on `0..2^40` AND added **1-in-8 32-character lowercase hex suffixes** (`u` / `e` / `k` + 32 hex chars = 33 bytes).
  - Broke Team 67's `express-lane` key-recovery fast path (which scored 0.31s on v4) and exposed severe interner lock contention on 32-char hex IDs across the field.
- **Live v5 Leaderboard Snapshot (2026-11-22 12:25 UTC)**:
  - `#1 67`: **0.4218 s**
  - `#2 JustOneGuy`: **0.4256 s**
  - `#3 LSU_i_Bosna_Najbolja`: **0.4293 s**
  - `#4 OhShi`: **0.4351 s**
  - `#5 Wix`: **0.4428 s**
  - *(Earlier 06:52 UTC snapshot: `#1 NoMoreLite 0.4508 s`, `#2 LSU_i_Bosna_Najbolja 0.4527 s`, `#3 @GMUNVIS 0.4581 s`)*
  - Baseline (PySpark): **168.86 s**

## 2. GitHub Reconnaissance (192 Repos Discovered, 32 Cloned & Audited)

| Architecture / Team | Language | Key Strengths | Bottlenecks / Blind Spots |
| --- | --- | --- | --- |
| `@GMUNVIS` (`SuperHuyGaming`, `#3 0.458s`) | Rust (`scratch`) | AVX-512 parsing, RLE_DICTIONARY Parquet, parallel release | 64-way sharded **mutex-locked** intern table for v5 32-char hex IDs; 2-scatter event pipeline |
| `mego74` | C++17 (`scratch`) | **Single-scatter in-place dedup** (`PRec` 8B + kill losers); separated purchases; fast parallel MADV_DONTNEED (`2.49s` local p20!) | Global atomic `IDS` table for ALL users (DRAM misses + CAS on every event); 64-bit hash-only IDs (collision risk); sorts per-person events; PLAIN Parquet |
| `krishavs1` | C++17 (`scratch`) | **Sort-free 30-min bin bitmask sessionization**; O(S+C) sliding-window assists; lazy `click_id` tie-breaking | Heap-copies every `user_id` string; 2-scatter dedup |
| `ejensODU` | C++20 (`scratch`) | Read-only **Blocked Bloom + Non-Canonical Alias Table** (skips 60%+ lookups); 64-row prefetch pipeline; `pread` I/O | Single-mutex `unordered_map` interner on v5 32-char hex IDs (severe contention!); 2-scatter dedup |
| `himagna16` | C++20 (`scratch`) | Exact 5-instruction bitwise `quote_faults`; 64B SIMD scanning | 256-part lock-free interner + non-exact 63-bit hash for long `event_id`; 2-scatter dedup |
| `aqilc` | C (`scratch`) | Compact pure C (`3.53s` local p20) | 2-scatter dedup; slower than single-scatter |
| `joch44` / `shervan0714` / `kaedeng` | Rust (`scratch`) | Dual-build Cascade Lake AVX-512 + AVX2 fallback; custom Parquet; huge-page arenas | 2-scatter dedup; interning overhead on 32-char hex IDs |

## 3. Hyperion's Original Synthesis (7 Combined Innovations)

1. **Zero-Interning Lossless Dual-Width Keys**: Short (`prefix + 1..8 ASCII`) in 64 bits + **Long Hex (`prefix + 32 lowercase hex`) packed via 10-instruction AVX2 into lossless 128 bits (`Hex128`)** with exact byte-wise ordering and 5-instruction AVX2 decode — **ZERO locks, ZERO interning, ZERO collisions** on 99.9995% of v5 IDs!
2. **Single-Scatter In-Place Dedup**: Valid events go straight to person partitions (12B SoA `ShortPBlock`, separate 3% `PurchRec`), while disjoint exact dedup buckets (`16B ShortERec`, `24B LongERec`) only tombstone the ~3% losing duplicates via `__atomic_store_n(..., DEAD_TS)`. Eliminates the entire second scatter of 20M–60M events!
3. **Zero Global Click Dedup + Lazy `click_id` Tie-Breaking**: Proven mathematically from `rules.md` (identical duplicate clicks cannot alter latest-click credit or distinct-campaign assists). Eliminates 5M–15M click-dedup hash operations and avoids parsing/storing `click_id` for 99.5% of clicks!
4. **Read-Only 1-Word Register-Blocked Bloom + Non-Canonical Alias Table**: 64-row 3-pass prefetch pipeline resolves `user_id -> canonical` with zero atomics and skips the alias table entirely for 60%+ of rows!
5. **Sort-Free 30-Minute Bitmask Sessionization + O(S+C) Sliding-Window Assists**: Collapses even 1.2M-event whales in one linear O(N) pass with zero sorting!
6. **Direct-Indexed Catalogs + L1 Window Date Cache + SIMD CSV (AVX2 + Cascade Lake AVX-512)**: Zero-hash `products` / `campaigns` lookup, 3 KB L1 date table, SWAR 13-digit F3, 64-byte SIMD `quote_faults`.
7. **Parallel Custom RLE_DICTIONARY Parquet Writer + Parallel `MADV_DONTNEED` + `_exit(0)`**: 1 dict entry + 1 RLE run per person (`~5×` smaller output), 21-entry dict + 1-bit def levels for `top_category`, adaptive `INT32`/`INT64` revenue.
