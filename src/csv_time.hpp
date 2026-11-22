#pragma once

#include "common.hpp"
#include <immintrin.h>
#include <limits>
#include <string_view>

namespace hyp {

inline constexpr i64 INVALID_TS = std::numeric_limits<i64>::min();

// ============================================================================
// Exact Gregorian Calendar & Zero-Hash SWAR Timestamp Parser
// ============================================================================

inline bool is_leap_year(i64 y) {
    return (y % 4 == 0) && (y % 100 != 0 || y % 400 == 0);
}

inline int days_in_month(i64 y, int m) {
    static constexpr int D[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return D[m] + (m == 2 && is_leap_year(y));
}

inline i64 days_from_civil(i64 y, int m, int d) {
    y -= (m <= 2);
    const i64 era = (y >= 0 ? y : y - 399) / 400;
    const u32 yoe = u32(y - era * 400);
    const u32 doy = (153u * u32(m + (m > 2 ? -3 : 9)) + 2u) / 5u + u32(d) - 1u;
    const u32 doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097LL + i64(doe) - 719468LL;
}

inline i64 floor_div_day(i64 ms) {
    i64 q = ms / DAY_MS;
    i64 r = ms % DAY_MS;
    return r < 0 ? (q - 1) : q;
}

static inline u64 load64u(const u8* p) {
    u64 v;
    std::memcpy(&v, p, 8);
    return v;
}

static inline bool is_ascii_dig(u8 c) { return u32(c - '0') < 10u; }

// SWAR 8-digit validation + conversion (first ASCII digit in lowest byte)
static inline bool all_digits8(u64 v) {
    const u64 t = v ^ 0x3030303030303030ULL;
    return ((t | (t + 0x0606060606060606ULL)) & 0xF0F0F0F0F0F0F0F0ULL) == 0;
}

static inline u64 swar8_to_u64(u64 v) {
    v -= 0x3030303030303030ULL;
    v = (v * 10ULL) + (v >> 8);
    v = (((v & 0x000000FF000000FFULL) * (100ULL + (1000000ULL << 32))) +
         (((v >> 16) & 0x000000FF000000FFULL) * (1ULL + (10000ULL << 32)))) >> 32;
    return v;
}

// Fast 13-digit F3 parser using 2 overlapping 8-byte SWAR loads!
static inline i64 parse_f3_13(const u8* p) {
    u64 a = load64u(p);
    u64 b = load64u(p + 5);
    if (__builtin_expect(all_digits8(a) & all_digits8(b), 1)) {
        return i64(swar8_to_u64(a) * 100000ULL + (swar8_to_u64(b) % 100000ULL));
    }
    return INVALID_TS;
}

inline i64 parse_f3_general(const u8* p, size_t n) {
    if (n == 0) return INVALID_TS;
    size_t i = 0;
    while (i < n && p[i] == '0') ++i;
    if (i == n) return 0;
    size_t rem = n - i;
    if (rem > 18) {
        for (; i < n; ++i) {
            if (!is_ascii_dig(p[i])) return INVALID_TS;
        }
        return std::numeric_limits<i64>::max();
    }
    i64 v = 0;
    for (; i < n; ++i) {
        u32 d = u32(p[i] - '0');
        if (d >= 10u) return INVALID_TS;
        v = v * 10 + i64(d);
    }
    return v;
}

// Precomputed month-start & days-in-month LUT for years 1970..2100 (zero hashing, 1 array load!)
struct FastTime {
    static constexpr int YBASE = 1970;
    static constexpr int YSPAN = 132; // 1970..2101
    i32 mstart[YSPAN * 13]{};
    u8  mdim[YSPAN * 13]{};

    void init(i64 /*ws*/, i64 /*we*/) {
        for (int yi = 0; yi < YSPAN; ++yi) {
            int y = YBASE + yi;
            for (int m = 1; m <= 12; ++m) {
                mstart[yi * 13 + m] = i32(days_from_civil(y, m, 1));
                mdim[yi * 13 + m] = u8(days_in_month(y, m));
            }
        }
    }

    // Exact SWAR validator + parser for 19-byte "YYYY-MM-DDTHH:MM:SS"
    // Caller guarantees at least 24 readable bytes at p (MFile has 4KB zero guard past EOF).
    inline bool parse_dt19_sec(const u8* s, i64& secs) const {
        const u64 a = load64u(s);
        const u64 b = load64u(s + 8);
        const u64 c = load64u(s + 16);
        constexpr u64 SA = 0xFF0000FF00000000ULL;
        constexpr u64 VA = (u64('-') << 32) | (u64('-') << 56);
        constexpr u64 SB = 0x0000FF0000FF0000ULL;
        constexpr u64 VB = (u64('T') << 16) | (u64(':') << 40);
        if (__builtin_expect((((a & SA) ^ VA) | ((b & SB) ^ VB) | ((c & 0xFFu) ^ u64(':'))) != 0, 0))
            return false;

        constexpr u64 Z   = 0x3030303030303030ULL;
        constexpr u64 SIX = 0x0606060606060606ULL;
        constexpr u64 HI  = 0xF0F0F0F0F0F0F0F0ULL;
        constexpr u64 DA  = 0x00FFFF00FFFFFFFFULL; // bytes 0,1,2,3 (YYYY) and 5,6 (MM)
        constexpr u64 DB  = 0xFFFF00FFFF00FFFFULL; // bytes 8,9 (DD), 11,12 (HH), 14,15 (MI)
        constexpr u64 DC  = 0x0000000000FFFF00ULL; // bytes 17,18 (SS)
        const u64 ta = (a ^ Z) & DA;
        const u64 tb = (b ^ Z) & DB;
        const u64 tc = (c ^ Z) & DC;
        if (__builtin_expect((((ta | (ta + SIX)) & HI & DA) |
                              ((tb | (tb + SIX)) & HI & DB) |
                              ((tc | (tc + SIX)) & HI & DC)) != 0, 0))
            return false;

        const u64 pa = ta * 10ULL + (ta >> 8);
        const u64 pb = tb * 10ULL + (tb >> 8);
        const u64 pc = tc * 10ULL + (tc >> 8);
        const u32 y  = u32(pa & 0xFFu) * 100u + u32((pa >> 16) & 0xFFu);
        const u32 mo = u32((pa >> 40) & 0xFFu);
        const u32 dd = u32(pb & 0xFFu);
        const u32 hh = u32((pb >> 24) & 0xFFu);
        const u32 mi = u32((pb >> 48) & 0xFFu);
        const u32 ss = u32((pc >> 8) & 0xFFu);
        if (__builtin_expect((mo - 1u > 11u) | (dd == 0u) | (hh > 23u) | (mi > 59u) | (ss > 59u), 0))
            return false;

        i64 days;
        u32 yi = y - u32(YBASE);
        if (__builtin_expect(yi < u32(YSPAN), 1)) {
            if (__builtin_expect(dd > mdim[yi * 13u + mo], 0)) return false;
            days = i64(mstart[yi * 13u + mo]) + i64(dd) - 1LL;
        } else {
            if (dd > u32(days_in_month(y, int(mo)))) return false;
            days = days_from_civil(y, int(mo), int(dd));
        }
        secs = days * 86400LL + i64(hh * 3600u + mi * 60u + ss);
        return true;
    }

    // Ultra-fast F1 parser for ingest_ts (rules.md guarantees ingest_ts is ALWAYS valid F1!)
    inline i64 parse_ingest_sec(const u8* s, u32 n) const {
        if (__builtin_expect(n != 20 || s[19] != 'Z', 0)) return INVALID_TS;
        constexpr u64 Z  = 0x3030303030303030ULL;
        const u64 ta = (load64u(s)      ^ Z) & 0x00FFFF00FFFFFFFFULL;
        const u64 tb = (load64u(s + 8)  ^ Z) & 0xFFFF00FFFF00FFFFULL;
        const u64 tc = (load64u(s + 16) ^ Z) & 0x0000000000FFFF00ULL;
        const u64 pa = ta * 10ULL + (ta >> 8);
        const u64 pb = tb * 10ULL + (tb >> 8);
        const u64 pc = tc * 10ULL + (tc >> 8);
        const u32 y  = u32(pa & 0xFFu) * 100u + u32((pa >> 16) & 0xFFu);
        const u32 mo = u32((pa >> 40) & 0xFFu);
        const u32 yi = y - u32(YBASE);
        i64 days = (__builtin_expect(yi < u32(YSPAN) && (mo - 1u) < 12u, 1))
                   ? i64(mstart[yi * 13u + mo])
                   : days_from_civil(y, mo ? int(mo) : 1, 1);
        days += i64(pb & 0xFFu) - 1LL;
        return days * 86400LL + i64(((pb >> 24) & 0xFFu) * 3600u +
                                    ((pb >> 48) & 0xFFu) * 60u +
                                    ((pc >> 8) & 0xFFu));
    }

    inline i64 parse(const u8* s, size_t n) const {
        if (__builtin_expect(n == 20, 1)) {
            i64 secs;
            if (__builtin_expect(s[19] == 'Z' && parse_dt19_sec(s, secs), 1)) {
                return secs * 1000LL;
            }
            return INVALID_TS;
        }
        if (__builtin_expect(n == 13, 1)) {
            return parse_f3_13(s);
        }
        if (__builtin_expect(n == 25, 1)) {
            u8 sgn = s[19];
            if (__builtin_expect((sgn != '+' && sgn != '-') || s[22] != ':', 0)) return INVALID_TS;
            u32 a = u32(s[20] - '0'), b = u32(s[21] - '0');
            u32 c = u32(s[23] - '0'), d = u32(s[24] - '0');
            if (__builtin_expect((a > 9u) | (b > 9u) | (c > 9u) | (d > 9u), 0)) return INVALID_TS;
            u32 oh = a * 10u + b, om = c * 10u + d;
            if (__builtin_expect(oh > 23u || om > 59u, 0)) return INVALID_TS;
            i64 secs;
            if (__builtin_expect(!parse_dt19_sec(s, secs), 0)) return INVALID_TS;
            i64 off = i64(oh * 3600u + om * 60u);
            if (sgn == '-') off = -off;
            return (secs - off) * 1000LL;
        }
        return parse_f3_general(s, n);
    }
};

inline i64 parse_ts_general(const u8* p, size_t n) {
    static FastTime ft = []() { FastTime t; t.init(0, 0); return t; }();
    return ft.parse(p, n);
}

static inline bool parse_price_cents(const u8* p, size_t n, u32& out) {
    if (__builtin_expect(n == 0, 0)) return false;
    u64 v = 0;
    for (size_t i = 0; i < n; ++i) {
        u32 d = u32(p[i] - '0');
        if (__builtin_expect(d >= 10u, 0)) return false;
        v = v * 10u + d;
        if (__builtin_expect(v > 10'000'000ULL, 0)) return false;
    }
    if (__builtin_expect(v == 0, 0)) return false;
    out = u32(v);
    return true;
}

// ============================================================================
// Zero-Overlap 64-Byte Aligned Streaming CSV Scanner (2-Mask AVX2 / AVX-512)
// - Classifies each 64-byte aligned block ONCE (2 masks: sep=','|'\n', quo='"')
// - Validates quotes for the entire line branchlessly via cumulative quote count
// - Zero intermediate buffers, zero memchr on valid lines, handles any v5 line length!
// ============================================================================

struct Field {
    const u8* p;
    u32 n;
};

template <int NF>
static inline bool split_exact(const u8* a, const u8* end, Field out[NF]) {
    if (a >= end) return false;
    int idx = 0;
    const u8* cur = a;
    for (;;) {
        const u8* comma = static_cast<const u8*>(std::memchr(cur, ',', size_t(end - cur)));
        const u8* stop = comma ? comma : end;
        if (idx >= NF) return false;
        const u8* s = cur;
        const u8* e = stop;
        if (std::memchr(cur, '"', size_t(stop - cur))) {
            if (stop - cur < 2 || cur[0] != '"' || stop[-1] != '"') return false;
            if (std::memchr(cur + 1, '"', size_t(stop - cur - 2))) return false;
            s = cur + 1;
            e = stop - 1;
        }
        out[idx].p = s;
        out[idx].n = u32(e - s);
        idx++;
        if (!comma) break;
        cur = comma + 1;
    }
    return idx == NF;
}

static inline void block_masks64(const u8* p, u64& sep, u64& quo) {
#if defined(__AVX512BW__)
    __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(p));
    u64 cm = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(',')));
    u64 nl = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('\n')));
    sep = cm | nl;
    quo = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('"')));
#else
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + 32));
    const __m256i C = _mm256_set1_epi8(',');
    const __m256i N = _mm256_set1_epi8('\n');
    const __m256i Q = _mm256_set1_epi8('"');
    u64 sa = u32(_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(a, C), _mm256_cmpeq_epi8(a, N))));
    u64 sb = u32(_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(b, C), _mm256_cmpeq_epi8(b, N))));
    u64 qa = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, Q)));
    u64 qb = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, Q)));
    sep = sa | (sb << 32);
    quo = qa | (qb << 32);
#endif
}

struct StreamScanner {
    const u8* base;
    size_t blk;   // current 64-byte aligned offset
    size_t limit; // safe end (last '\n' + 1 inside chunk)
    u64 sm;       // remaining separator bits in current 64B block
    u64 qm;       // all quote bits in current 64B block
    u64 cumq;     // cumulative quotes in blocks strictly before blk

    inline void init(const u8* b, size_t pos, size_t lim) {
        base = b;
        limit = lim;
        blk = pos & ~size_t(63);
        u64 s, q;
        block_masks64(base + blk, s, q);
        sm = s & (~0ULL << (pos & 63u));
        qm = q;
        cumq = 0;
    }

    inline size_t next() {
        while (__builtin_expect(sm == 0, 0)) {
            cumq += u64(__builtin_popcountll(qm));
            blk += 64;
            if (__builtin_expect(blk >= limit, 0)) return limit;
            block_masks64(base + blk, sm, qm);
        }
        size_t p = blk + size_t(__builtin_ctzll(sm));
        sm &= sm - 1ULL;
        return p;
    }

    inline u64 qbefore(size_t pos) const {
        size_t k = pos - blk;
        u64 m = (k >= 64) ? ~0ULL : ((1ULL << k) - 1ULL);
        return cumq + u64(__builtin_popcountll(qm & m));
    }
};

template <int NF, class Callback>
static inline void scan_csv(const u8* base, size_t start, size_t end, Callback&& cb) {
    if (start >= end) return;
    // Find last newline in [start, end) so the hot streaming loop ALWAYS terminates on '\n'
    size_t fast_end = end;
    while (fast_end > start && base[fast_end - 1] != '\n') --fast_end;

    Field fl[NF];
    if (fast_end > start) {
        StreamScanner sc;
        sc.init(base, start, fast_end);
        size_t ls = start;
        while (ls < fast_end) {
            const u64 q0 = sc.qbefore(ls);
            size_t fs = ls;
            u32 expq = 0;
            u32 bad = 0;
            bool ok = true;
            for (int i = 0; i < NF; ++i) {
                size_t p = sc.next();
                u8 c = base[p];
                if (i < NF - 1) {
                    if (__builtin_expect(c == '\n', 0)) {
                        ls = p + 1;
                        ok = false;
                        break;
                    }
                } else {
                    if (__builtin_expect(c != '\n', 0)) {
                        do { p = sc.next(); } while (p < fast_end && base[p] != '\n');
                        ls = p + 1;
                        ok = false;
                        break;
                    }
                }
                u32 len = u32(p - fs);
                u32 isq = (len > 0 && base[fs] == '"') ? 1u : 0u;
                bad |= isq & ((len < 2u) | (base[p - 1] != '"'));
                fl[i].p = base + fs + isq;
                fl[i].n = len - 2u * isq;
                expq += 2u * isq;
                fs = p + 1;
            }
            if (__builtin_expect(ok, 1)) {
                ls = fs;
                // Check exact quote count across the line: equals 2 * (# quoted fields) iff zero bad quotes!
                if (__builtin_expect(bad == 0 && (sc.qbefore(fs - 1) - q0) == u64(expq), 1)) {
                    cb(fl);
                }
            }
        }
    }

    // Handle trailing partial line without '\n' at EOF (if any)
    if (fast_end < end) {
        if (split_exact<NF>(base + fast_end, base + end, fl)) {
            cb(fl);
        }
    }
}

} // namespace hyp
