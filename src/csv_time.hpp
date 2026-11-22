#pragma once

#include "common.hpp"
#include <immintrin.h>
#include <limits>
#include <string_view>

namespace HYP_NS {

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

inline void civil_from_days(i64 z, i64& y, int& m, int& d) {
    z += 719468LL;
    const i64 era = (z >= 0 ? z : z - 146096LL) / 146097LL;
    const u32 doe = u32(z - era * 146097LL);
    const u32 yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
    y = i64(yoe) + era * 400LL;
    const u32 doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
    const u32 mp = (5u * doy + 2u) / 153u;
    d = int(doy - (153u * mp + 2u) / 5u + 1u);
    m = int(mp < 10u ? mp + 3u : mp - 9u);
    y += (m <= 2);
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

// Fast 13-digit F3 parser using 2 overlapping 8-byte SWAR loads (zero division/modulo!)
static inline i64 parse_f3_13(const u8* p) {
    u64 a = load64u(p);
    u64 b = load64u(p + 5);
    if (__builtin_expect((int(all_digits8(a)) & int(all_digits8(b))) != 0, 1)) {
        u64 b5 = (b & 0xFFFFFFFFFF000000ULL) | 0x0000000000303030ULL;
        return i64(swar8_to_u64(a) * 100000ULL + swar8_to_u64(b5));
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

// Precomputed 2-month prefix fast path + full 1970..2101 LUT (zero hashing!)
struct FastTime {
    static constexpr int YBASE = 1970;
    static constexpr int YSPAN = 132;
    u64 ym0_ascii = 0;
    u64 ym1_ascii = 0;
    i64 ym0_sec_base = 0; // (days_from_civil(y, m, 1) - 1) * 86400
    i64 ym1_sec_base = 0;
    u32 ym0_dim = 0;
    u32 ym1_dim = 0;

    i32 mstart[YSPAN * 13]{};
    u8  mdim[YSPAN * 13]{};

    static u64 make_ym_ascii(i64 y, int m) {
        if (y < 0 || y > 9999 || m < 1 || m > 12) return 0;
        u8 buf[8];
        buf[0] = u8('0' + (y / 1000) % 10);
        buf[1] = u8('0' + (y / 100) % 10);
        buf[2] = u8('0' + (y / 10) % 10);
        buf[3] = u8('0' + y % 10);
        buf[4] = '-';
        buf[5] = u8('0' + m / 10);
        buf[6] = u8('0' + m % 10);
        buf[7] = '-';
        return load64u(buf);
    }

    void init(i64 ws, i64 we) {
        for (int yi = 0; yi < YSPAN; ++yi) {
            int y = YBASE + yi;
            for (int m = 1; m <= 12; ++m) {
                mstart[yi * 13 + m] = i32(days_from_civil(y, m, 1));
                mdim[yi * 13 + m] = u8(days_in_month(y, m));
            }
        }
        if (we > ws) {
            i64 mid_day = floor_div_day(ws + (we - ws) / 2);
            i64 y0; int m0, d0;
            civil_from_days(mid_day, y0, m0, d0);
            ym0_ascii = make_ym_ascii(y0, m0);
            ym0_sec_base = (days_from_civil(y0, m0, 1) - 1LL) * 86400LL;
            ym0_dim = u32(days_in_month(y0, m0));

            // Pick secondary month (either start-of-click-window month or end-of-window month)
            i64 y1; int m1, d1;
            civil_from_days(floor_div_day(ws) - 7, y1, m1, d1);
            if (y1 == y0 && m1 == m0) {
                civil_from_days(floor_div_day(we) + 1, y1, m1, d1);
            }
            if (y1 != y0 || m1 != m0) {
                ym1_ascii = make_ym_ascii(y1, m1);
                ym1_sec_base = (days_from_civil(y1, m1, 1) - 1LL) * 86400LL;
                ym1_dim = u32(days_in_month(y1, m1));
            }
        }
    }

    // Exact SWAR validator + parser for 19-byte "YYYY-MM-DDTHH:MM:SS"
    inline bool parse_dt19_sec(const u8* s, i64& secs) const {
        const u64 a = load64u(s);
        const u64 b = load64u(s + 8);

        constexpr u64 SB = 0x0000FF0000FF0000ULL;
        constexpr u64 VB = (u64('T') << 16) | (u64(':') << 40);
        constexpr u64 Z   = 0x3030303030303030ULL;
        constexpr u64 SIX = 0x0606060606060606ULL;
        constexpr u64 HI  = 0xF0F0F0F0F0F0F0F0ULL;
        constexpr u64 DB  = 0xFFFF00FFFF00FFFFULL; // DD, HH, MI
        constexpr u64 DC  = 0x0000000000FFFF00ULL; // SS

        // Hot path: "YYYY-MM-" matches precomputed window month (skips YYYY, MM, '-', '-' & mstart LUT!)
        if (__builtin_expect(a == ym0_ascii || a == ym1_ascii, 1)) {
            if (__builtin_expect((((b & SB) ^ VB) | (u64(s[16]) ^ u64(':'))) != 0, 0))
                return false;
            const u64 tb = (b ^ Z) & DB;
            const u32 s1 = u32(s[17] - '0'), s2 = u32(s[18] - '0');
            if (__builtin_expect((((tb | (tb + SIX)) & HI & DB) != 0) | (s1 > 5u) | (s2 > 9u), 0))
                return false;
            const u64 pb = tb * 10ULL + (tb >> 8);
            const u32 dd = u32(pb & 0xFFu);
            const u32 hh = u32((pb >> 24) & 0xFFu);
            const u32 mi = u32((pb >> 48) & 0xFFu);
            const u32 ss = s1 * 10u + s2;
            const bool is0 = (a == ym0_ascii);
            const u32 dim = is0 ? ym0_dim : ym1_dim;
            if (__builtin_expect((dd == 0u) | (dd > dim) | (hh > 23u) | (mi > 59u), 0))
                return false;
            const i64 base = is0 ? ym0_sec_base : ym1_sec_base;
            secs = base + i64(dd) * 86400LL + i64(hh * 3600u + mi * 60u + ss);
            return true;
        }

        // General path: any year/month
        const u64 c = load64u(s + 16);
        constexpr u64 SA = 0xFF0000FF00000000ULL;
        constexpr u64 VA = (u64('-') << 32) | (u64('-') << 56);
        constexpr u64 DA = 0x00FFFF00FFFFFFFFULL;
        if (__builtin_expect((((a & SA) ^ VA) | ((b & SB) ^ VB) | ((c & 0xFFu) ^ u64(':'))) != 0, 0))
            return false;
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
        constexpr u64 Z = 0x3030303030303030ULL;
        const u64 a  = load64u(s);
        const u64 tb = (load64u(s + 8) ^ Z) & 0xFFFF00FFFF00FFFFULL;
        const u64 pb = tb * 10ULL + (tb >> 8);
        const u32 ss = u32(s[17] - '0') * 10u + u32(s[18] - '0');
        const i64 d_hms = i64(pb & 0xFFu) * 86400LL +
                          i64(((pb >> 24) & 0xFFu) * 3600u +
                              ((pb >> 48) & 0xFFu) * 60u + ss);
        if (__builtin_expect(a == ym0_ascii, 1)) return ym0_sec_base + d_hms;
        if (__builtin_expect(a == ym1_ascii, 1)) return ym1_sec_base + d_hms;
        const u64 ta = (a ^ Z) & 0x00FFFF00FFFFFFFFULL;
        const u64 pa = ta * 10ULL + (ta >> 8);
        const u32 y  = u32(pa & 0xFFu) * 100u + u32((pa >> 16) & 0xFFu);
        const u32 mo = u32((pa >> 40) & 0xFFu);
        const u32 yi = y - u32(YBASE);
        i64 days = (__builtin_expect(yi < u32(YSPAN) && (mo - 1u) < 12u, 1))
                   ? i64(mstart[yi * 13u + mo])
                   : days_from_civil(y, mo ? int(mo) : 1, 1);
        return (days - 1LL) * 86400LL + d_hms;
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
// Zero-Overlap 64-Byte Aligned Streaming CSV Scanner (AVX2 / AVX-512)
// - Classifies each 64B aligned block ONCE (2 masks: sep=','|'\n', quo='"')
// - Zero-quote fast path: skips ALL quote byte loads on unquoted lines!
// - Software prefetch 8 cache lines ahead for linear streaming bandwidth
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

static inline void block_masks64(const u8* p, u64& sep, u64& nlm, u64& quo) {
#if defined(__AVX512BW__)
    __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(p));
    u64 cm = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(',')));
    nlm = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('\n')));
    sep = cm | nlm;
    quo = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('"')));
#else
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + 32));
    const __m256i C = _mm256_set1_epi8(',');
    const __m256i N = _mm256_set1_epi8('\n');
    const __m256i Q = _mm256_set1_epi8('"');
    u64 ca = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, C)));
    u64 cb = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, C)));
    u64 na = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, N)));
    u64 nb = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, N)));
    u64 qa = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, Q)));
    u64 qb = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, Q)));
    nlm = na | (nb << 32);
    sep = (ca | (cb << 32)) | nlm;
    quo = qa | (qb << 32);
#endif
}

struct StreamScanner {
    const u8* base;
    size_t blk;
    size_t limit;
    u64 sm;
    u64 nlm;
    u64 qm;
    u64 cumq;

    inline void init(const u8* b, size_t pos, size_t lim) {
        base = b;
        limit = lim;
        blk = pos & ~size_t(63);
        u64 s, n, q;
        block_masks64(base + blk, s, n, q);
        sm = s & (~0ULL << (pos & 63u));
        nlm = n;
        qm = q;
        cumq = 0;
    }

    inline size_t next(bool& is_nl) {
        while (__builtin_expect(sm == 0, 0)) {
            cumq += u64(__builtin_popcountll(qm));
            blk += 64;
            if (__builtin_expect(blk >= limit, 0)) { is_nl = true; return limit; }
            __builtin_prefetch(base + blk + 512, 0, 0);
            block_masks64(base + blk, sm, nlm, qm);
        }
        u64 bit = sm & -sm;
        is_nl = (nlm & bit) != 0;
        size_t p = blk + size_t(__builtin_ctzll(sm));
        sm ^= bit;
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
    size_t fast_end = end;
    while (fast_end > start && base[fast_end - 1] != '\n') --fast_end;

    Field fl[NF];
    if (fast_end > start) {
        StreamScanner sc;
        sc.init(base, start, fast_end);
        size_t ls = start;
        u64 q0 = sc.qbefore(ls);
        while (ls < fast_end) {
            size_t fs = ls;
            u32 expq = 0;
            u32 bad = 0;
            bool ok = true;
            #pragma GCC unroll 8
            for (int i = 0; i < NF; ++i) {
                bool is_nl;
                size_t p = sc.next(is_nl);
                if (i < NF - 1) {
                    if (__builtin_expect(is_nl, 0)) {
                        ls = p + 1;
                        q0 = sc.qbefore(ls);
                        ok = false;
                        break;
                    }
                } else {
                    if (__builtin_expect(!is_nl, 0)) {
                        do { p = sc.next(is_nl); } while (p < fast_end && !is_nl);
                        ls = p + 1;
                        q0 = sc.qbefore(ls);
                        ok = false;
                        break;
                    }
                }
                // 100% branchless quote stripping (zero mispredictions on 79% quoted lines!)
                u32 len = u32(p - fs);
                u32 isq = (base[fs] == '"') ? 1u : 0u;
                bad |= isq & ((len < 2u) | (base[p - isq] != '"'));
                fl[i].p = base + fs + isq;
                fl[i].n = len - 2u * isq;
                expq += 2u * isq;
                fs = p + 1;
            }
            if (__builtin_expect(ok, 1)) {
                ls = fs;
                u64 q1 = sc.qbefore(fs - 1);
                u64 dq = q1 - q0;
                q0 = q1;
                if (__builtin_expect(bad == 0 && dq == u64(expq), 1)) {
                    cb(fl);
                }
            }
        }
    }

    if (fast_end < end) {
        if (split_exact<NF>(base + fast_end, base + end, fl)) {
            cb(fl);
        }
    }
}

} // namespace HYP_NS
