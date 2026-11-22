#pragma once

#include "common.hpp"
#include <immintrin.h>
#include <limits>
#include <string_view>

namespace hyp {

inline constexpr i64 INVALID_TS = std::numeric_limits<i64>::min();

// ============================================================================
// Exact Gregorian Calendar & Window-Cached Fast Timestamp Parser
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

inline void civil_from_days(i64 z, int& y, int& m, int& d) {
    z += 719468LL;
    const i64 era = (z >= 0 ? z : z - 146096LL) / 146097LL;
    const u32 doe = u32(z - era * 146097LL);
    const u32 yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
    y = int(yoe) + int(era) * 400;
    const u32 doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
    const u32 mp = (5u * doy + 2u) / 153u;
    d = int(doy - (153u * mp + 2u) / 5u + 1u);
    m = int(mp + (mp < 10 ? 3 : -9));
    y += (m <= 2);
}

inline i64 floor_div_day(i64 ms) {
    i64 q = ms / DAY_MS;
    i64 r = ms % DAY_MS;
    return r < 0 ? (q - 1) : q;
}

static inline bool is_ascii_dig(u8 c) { return u32(c - '0') < 10u; }
static inline bool is_dig2(const u8* p) { return u32(p[0] - '0') < 10u && u32(p[1] - '0') < 10u; }
static inline u32 parse_dig2(const u8* p) { return u32(p[0] - '0') * 10u + u32(p[1] - '0'); }

inline i64 parse_civil_general(const u8* p) {
    if (!is_dig2(p) || !is_dig2(p + 2) || p[4] != '-' ||
        !is_dig2(p + 5) || p[7] != '-' || !is_dig2(p + 8) ||
        p[10] != 'T' || !is_dig2(p + 11) || p[13] != ':' ||
        !is_dig2(p + 14) || p[16] != ':' || !is_dig2(p + 17)) {
        return INVALID_TS;
    }
    int y = int(parse_dig2(p) * 100u + parse_dig2(p + 2));
    int m = int(parse_dig2(p + 5));
    int d = int(parse_dig2(p + 8));
    u32 h = parse_dig2(p + 11);
    u32 mi = parse_dig2(p + 14);
    u32 s = parse_dig2(p + 17);
    if (m < 1 || m > 12 || d < 1 || d > days_in_month(y, m) || h >= 24u || mi >= 60u || s >= 60u) {
        return INVALID_TS;
    }
    return days_from_civil(y, m, d) * DAY_MS + i64(h * 3600u + mi * 60u + s) * 1000LL;
}

inline i64 parse_iso_general(const u8* p, size_t n) {
    if (n == 20 && p[19] == 'Z') {
        return parse_civil_general(p);
    }
    if (n == 25 && (p[19] == '+' || p[19] == '-') && p[22] == ':' &&
        is_dig2(p + 20) && is_dig2(p + 23)) {
        u32 oh = parse_dig2(p + 20);
        u32 om = parse_dig2(p + 23);
        if (oh >= 24u || om >= 60u) return INVALID_TS;
        i64 local = parse_civil_general(p);
        if (local == INVALID_TS) return INVALID_TS;
        i64 off = i64(oh * 60u + om) * 60000LL;
        return (p[19] == '+') ? (local - off) : (local + off);
    }
    return INVALID_TS;
}

// SWAR 13-digit exact ASCII parser (common case for F3 epoch milliseconds in 2001..2286)
static inline i64 parse_f3_13(const u8* p) {
    u64 raw5, lo;
    std::memcpy(&raw5, p, 8);
    std::memcpy(&lo, p + 5, 8);
    u64 hi = ((raw5 & 0x000000FFFFFFFFFFULL) << 24) - 0x3030303030000000ULL;
    lo -= 0x3030303030303030ULL;
    u64 bad = ((hi | (hi + 0x0606060606000000ULL)) |
               (lo | (lo + 0x0606060606060606ULL))) & 0xF0F0F0F0F0F0F0F0ULL;
    if (__builtin_expect(bad != 0, 0)) return INVALID_TS;

    hi = (hi * 10 + (hi >> 8)) & 0x00FF00FF00FF00FFULL;
    hi = (hi * 100 + (hi >> 16)) & 0x0000FFFF0000FFFFULL;
    hi = (hi * 10000 + (hi >> 32)) & 0x00000000FFFFFFFFULL;

    lo = (lo * 10 + (lo >> 8)) & 0x00FF00FF00FF00FFULL;
    lo = (lo * 100 + (lo >> 16)) & 0x0000FFFF0000FFFFULL;
    lo = (lo * 10000 + (lo >> 32)) & 0x00000000FFFFFFFFULL;

    return i64(hi) * 100'000'000LL + i64(lo);
}

// General F3 parser: strictly ASCII digits '0'..'9', leading zeros allowed.
// Values overflowing i64 are valid numbers outside any real window -> return INVALID_TS_OUT_OF_RANGE.
inline i64 parse_f3_general(const u8* p, size_t n) {
    if (n == 0) return INVALID_TS;
    size_t i = 0;
    while (i < n && p[i] == '0') ++i;
    if (i == n) return 0; // "0" or "000" -> epoch 0
    size_t rem = n - i;
    if (rem > 18) {
        // Verify remaining chars are ASCII digits (not malformed, just huge/outside window)
        for (; i < n; ++i) {
            if (!is_ascii_dig(p[i])) return INVALID_TS;
        }
        return std::numeric_limits<i64>::max(); // strictly outside any window
    }
    i64 v = 0;
    for (; i < n; ++i) {
        u32 d = u32(p[i] - '0');
        if (d >= 10u) return INVALID_TS;
        v = v * 10 + i64(d);
    }
    return v;
}

inline i64 parse_ts_general(const u8* p, size_t n) {
    if (n == 0) return INVALID_TS;
    if (n == 13) {
        i64 v = parse_f3_13(p);
        if (v != INVALID_TS) return v;
        return INVALID_TS;
    }
    if (n == 20 && p[19] == 'Z') return parse_civil_general(p);
    if (n == 25 && (p[19] == '+' || p[19] == '-')) return parse_iso_general(p, n);
    return parse_f3_general(p, n);
}

// L1-resident window date cache: maps 10-byte "YYYY-MM-DD" to midnight UTC ms
struct FastTime {
    struct Entry {
        u64 d8;     // bytes 0..7 ("YYYY-MM-")
        u16 d2;     // bytes 8..9 ("DD")
        u16 used;
        i32 pad;
        i64 day_ms; // midnight UTC in ms
    };
    static constexpr u32 CAP = 128; // 128 * 24 B = 3 KB (100% L1 resident!)
    static constexpr u32 MASK = CAP - 1;
    Entry tab[CAP]{};
    i64 win_start = 0;
    i64 win_end = 0;
    i64 click_start = 0;

    static inline u32 slot_hash(u64 d8, u16 d2) {
        // Mix day digits (d2) and month/year digits (d8) into 0..127
        return u32(((d8 >> 40) ^ (u64(d2) * 0x9E3779B1ULL)) * 2654435761ULL >> 25) & MASK;
    }

    void init(i64 ws, i64 we) {
        win_start = ws;
        win_end = we;
        click_start = ws - WEEK_MS;
        std::memset(tab, 0, sizeof(tab));
        i64 d0 = floor_div_day(ws) - 10;
        i64 d1 = floor_div_day(we) + 10;
        if (d1 - d0 > 96) d1 = d0 + 96; // keep load factor <= 0.75
        for (i64 day = d0; day <= d1; ++day) {
            int y, m, d;
            civil_from_days(day, y, m, d);
            if (y < 0 || y > 9999) continue;
            char buf[12];
            std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d);
            u64 d8;
            u16 d2;
            std::memcpy(&d8, buf, 8);
            std::memcpy(&d2, buf + 8, 2);
            u32 idx = slot_hash(d8, d2);
            while (tab[idx].used) idx = (idx + 1) & MASK;
            tab[idx] = {d8, d2, 1, 0, day * DAY_MS};
        }
    }

    inline i64 parse_hms(const u8* p) const {
        if (__builtin_expect(p[0] != 'T' || p[3] != ':' || p[6] != ':', 0)) return INVALID_TS;
        u32 h0 = u32(p[1] - '0'), h1 = u32(p[2] - '0');
        u32 m0 = u32(p[4] - '0'), m1 = u32(p[5] - '0');
        u32 s0 = u32(p[7] - '0'), s1 = u32(p[8] - '0');
        if (__builtin_expect((h0 | h1 | m0 | m1 | s0 | s1) >= 10u, 0)) {
            if (h0 >= 10u || h1 >= 10u || m0 >= 10u || m1 >= 10u || s0 >= 10u || s1 >= 10u)
                return INVALID_TS;
        }
        u32 h = h0 * 10u + h1;
        u32 mi = m0 * 10u + m1;
        u32 s = s0 * 10u + s1;
        if (__builtin_expect(h >= 24u || mi >= 60u || s >= 60u, 0)) return INVALID_TS;
        return i64(h * 3600u + mi * 60u + s) * 1000LL;
    }

    inline i64 lookup_date_ms(const u8* p) const {
        u64 d8;
        u16 d2;
        std::memcpy(&d8, p, 8);
        std::memcpy(&d2, p + 8, 2);
        u32 idx = slot_hash(d8, d2);
        for (;;) {
            const Entry& e = tab[idx];
            if (__builtin_expect(!e.used, 0)) return INVALID_TS;
            if (__builtin_expect(e.d8 == d8 && e.d2 == d2, 1)) return e.day_ms;
            idx = (idx + 1) & MASK;
        }
    }

    // Fast F1-only parser (used for ingest_ts, which rules.md guarantees is always valid F1)
    inline i64 parse_f1(const u8* p, size_t n) const {
        if (__builtin_expect(n != 20 || p[19] != 'Z', 0)) return INVALID_TS;
        i64 dms = lookup_date_ms(p);
        if (__builtin_expect(dms != INVALID_TS, 1)) {
            i64 hms = parse_hms(p + 10);
            if (__builtin_expect(hms != INVALID_TS, 1)) return dms + hms;
            return INVALID_TS;
        }
        return parse_civil_general(p);
    }

    // Fast general timestamp parser (F1, F2, F3)
    inline i64 parse(const u8* p, size_t n) const {
        if (__builtin_expect(n == 20, 1)) {
            if (__builtin_expect(p[19] != 'Z', 0)) return INVALID_TS;
            i64 dms = lookup_date_ms(p);
            if (__builtin_expect(dms != INVALID_TS, 1)) {
                i64 hms = parse_hms(p + 10);
                return hms != INVALID_TS ? (dms + hms) : INVALID_TS;
            }
            return parse_civil_general(p);
        }
        if (__builtin_expect(n == 13, 1)) {
            return parse_f3_13(p);
        }
        if (__builtin_expect(n == 25, 1)) {
            u8 sgn = p[19];
            if (__builtin_expect((sgn != '+' && sgn != '-') || p[22] != ':', 0)) return INVALID_TS;
            if (__builtin_expect(!is_dig2(p + 20) || !is_dig2(p + 23), 0)) return INVALID_TS;
            u32 oh = parse_dig2(p + 20);
            u32 om = parse_dig2(p + 23);
            if (__builtin_expect(oh >= 24u || om >= 60u, 0)) return INVALID_TS;
            i64 dms = lookup_date_ms(p);
            i64 local;
            if (__builtin_expect(dms != INVALID_TS, 1)) {
                i64 hms = parse_hms(p + 10);
                if (__builtin_expect(hms == INVALID_TS, 0)) return INVALID_TS;
                local = dms + hms;
            } else {
                local = parse_civil_general(p);
                if (local == INVALID_TS) return INVALID_TS;
            }
            i64 off = i64(oh * 60u + om) * 60000LL;
            return (sgn == '+') ? (local - off) : (local + off);
        }
        return parse_f3_general(p, n);
    }
};

// Exact ASCII price_cents parser for purchases: digits only, 1 .. 10,000,000 (leading zeros ok)
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
// Exact CSV Dialect Scanner (SIMD 64-byte AVX2 + Runtime AVX-512 Dispatch)
// ============================================================================

struct Field {
    const u8* p;
    u32 n;
};

// Exact byte-by-byte RFC-subset parser matching rules.md CSV dialect 100%:
// - Only '\n' ends a line (already stripped; [a, end) contains no '\n')
// - Empty line -> malformed
// - Every comma separates fields; quotes only allowed wrapping whole field
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

struct Mask64 {
    u64 comma;
    u64 nl;
    u64 quote;
};

static inline Mask64 classify64_avx2(const u8* p) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    auto eq = [&](char c) -> u64 {
        __m256i v = _mm256_set1_epi8(c);
        u32 lo = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, v)));
        u32 hi = u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, v)));
        return u64(lo) | (u64(hi) << 32);
    };
    return {eq(','), eq('\n'), eq('"')};
}

#if defined(__AVX512BW__)
static inline Mask64 classify64(const u8* p) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    u64 cm = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(',')));
    u64 nl = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('\n')));
    u64 qt = u64(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('"')));
    return {cm, nl, qt};
}
#else
static inline Mask64 classify64(const u8* p) {
    return classify64_avx2(p);
}
#endif

// Exact 5-instruction bitwise quote validator for a window starting at field start.
// Returns non-zero iff any quote violates the rules.md dialect before/at separators S.
static inline u64 quote_faults(u64 S, u64 Q) {
    const u64 first = (S << 1) | 1ULL;
    const u64 open = Q & first;
    const u64 other = Q & ~first;
    const u64 u = open << 1;
    const u64 t = other << 1;
    const u64 sum = ~S + open;
    return (t & ~S) | (u & S) | ((sum ^ (t | u)) & S);
}

// 64-byte SIMD line scanner with exact byte-by-byte fallback for long/malformed/boundary lines.
template <int NF, class Callback>
static inline void scan_csv(const u8* base, size_t start, size_t end, Callback&& cb) {
    auto classify = [](const u8* p) -> Mask64 { return classify64(p); };

    Field fl[NF];
    size_t pos = start;
    // Fast loop while we have >= 128 bytes ahead (MFile also has 4KB zero guard past EOF)
    while (pos < end) {
        const u8* p = base + pos;
        if constexpr (NF == 7) {
            // Event line: 7 fields, 6 commas + 1 newline.
            // Window A covers bytes [0..63] from line start.
            // Note: with v5 32-char hex IDs, event_id (33B) + user_id (9B) = 42B,
            // or event_id (33B) + user_id (33B) = 67B!
            Mask64 A = classify(p);
            u64 ca = A.comma;
            if (__builtin_expect(ca != 0, 1)) {
                u32 c1 = u32(__builtin_ctzll(ca)); ca &= ca - 1;
                if (__builtin_expect(ca != 0, 1)) {
                    u32 c2 = u32(__builtin_ctzll(ca)); ca &= ca - 1;
                    // Check no newline before c2 and no quote faults in first 2 fields
                    u64 maskA = (c2 == 63) ? ~0ULL : ((1ULL << (c2 + 1)) - 1ULL);
                    if (__builtin_expect((A.nl & maskA) == 0, 1)) {
                        u64 badA = quote_faults(A.comma | A.nl, A.quote) & maskA;
                        if (__builtin_expect(badA == 0, 1)) {
                            const u8* q = p + c2 + 1; // start of field 2 (ts)
                            Mask64 B = classify(q);
                            u64 commab = B.comma;
                            u64 nb = B.nl;
                            if (__builtin_expect(nb != 0, 1)) {
                                u32 e = u32(__builtin_ctzll(nb));
                                u64 maskB = (e == 63) ? ~0ULL : ((1ULL << (e + 1)) - 1ULL);
                                u64 cmB = commab & maskB;
                                // Need exactly 4 commas in [0..e) for fields 2,3,4,5,6
                                if (__builtin_expect(__builtin_popcountll(cmB) == 4, 1)) {
                                    u64 badB = quote_faults(B.comma | B.nl, B.quote) & maskB;
                                    if (__builtin_expect(badB == 0 && (pos + c2 + 1 + e) < end, 1)) {
                                        u32 c3 = u32(__builtin_ctzll(cmB)); cmB &= cmB - 1;
                                        u32 c4 = u32(__builtin_ctzll(cmB)); cmB &= cmB - 1;
                                        u32 c5 = u32(__builtin_ctzll(cmB)); cmB &= cmB - 1;
                                        u32 c6 = u32(__builtin_ctzll(cmB));
                                        u32 o0 = u32(A.quote & 1u);
                                        u32 o1 = u32((A.quote >> (c1 + 1)) & 1u);
                                        u32 o2 = u32(B.quote & 1u);
                                        u32 o3 = u32((B.quote >> (c3 + 1)) & 1u);
                                        u32 o4 = u32((B.quote >> (c4 + 1)) & 1u);
                                        u32 o5 = u32((B.quote >> (c5 + 1)) & 1u);
                                        u32 o6 = u32((B.quote >> (c6 + 1)) & 1u);
                                        fl[0] = {p + o0,           c1 - 2u * o0};
                                        fl[1] = {p + c1 + 1 + o1,  c2 - c1 - 1u - 2u * o1};
                                        fl[2] = {q + o2,           c3 - 2u * o2};
                                        fl[3] = {q + c3 + 1 + o3,  c4 - c3 - 1u - 2u * o3};
                                        fl[4] = {q + c4 + 1 + o4,  c5 - c4 - 1u - 2u * o4};
                                        fl[5] = {q + c5 + 1 + o5,  c6 - c5 - 1u - 2u * o5};
                                        fl[6] = {q + c6 + 1 + o6,  e - c6 - 1u - 2u * o6};
                                        pos += size_t(c2) + 1u + size_t(e) + 1u;
                                        cb(fl);
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if constexpr (NF == 4) {
            // Click line: 4 fields, 3 commas + 1 newline.
            // Fits in 1 window (64B) for 75% of clicks, or 2 windows if 32-char hex ID!
            Mask64 A = classify(p);
            u64 na = A.nl;
            if (__builtin_expect(na != 0, 1)) {
                u32 e = u32(__builtin_ctzll(na));
                u64 maskA = (e == 63) ? ~0ULL : ((1ULL << (e + 1)) - 1ULL);
                u64 cmA = A.comma & maskA;
                if (__builtin_expect(__builtin_popcountll(cmA) == 3, 1)) {
                    u64 badA = quote_faults(A.comma | A.nl, A.quote) & maskA;
                    if (__builtin_expect(badA == 0 && (pos + e) < end, 1)) {
                        u32 c1 = u32(__builtin_ctzll(cmA)); cmA &= cmA - 1;
                        u32 c2 = u32(__builtin_ctzll(cmA)); cmA &= cmA - 1;
                        u32 c3 = u32(__builtin_ctzll(cmA));
                        u32 o0 = u32(A.quote & 1u);
                        u32 o1 = u32((A.quote >> (c1 + 1)) & 1u);
                        u32 o2 = u32((A.quote >> (c2 + 1)) & 1u);
                        u32 o3 = u32((A.quote >> (c3 + 1)) & 1u);
                        fl[0] = {p + o0,           c1 - 2u * o0};
                        fl[1] = {p + c1 + 1 + o1,  c2 - c1 - 1u - 2u * o1};
                        fl[2] = {p + c2 + 1 + o2,  c3 - c2 - 1u - 2u * o2};
                        fl[3] = {p + c3 + 1 + o3,  e - c3 - 1u - 2u * o3};
                        pos += size_t(e) + 1u;
                        cb(fl);
                        continue;
                    }
                }
            } else if (__builtin_expect(A.comma != 0, 1)) {
                // 2-window click fast path (handles 32-char hex click_id or user_id!)
                u32 c1 = u32(__builtin_ctzll(A.comma));
                u64 maskA = (c1 == 63) ? ~0ULL : ((1ULL << (c1 + 1)) - 1ULL);
                if (__builtin_expect((quote_faults(A.comma | A.nl, A.quote) & maskA) == 0, 1)) {
                    const u8* q = p + c1 + 1;
                    Mask64 B = classify(q);
                    if (__builtin_expect(B.nl != 0, 1)) {
                        u32 e = u32(__builtin_ctzll(B.nl));
                        u64 maskB = (e == 63) ? ~0ULL : ((1ULL << (e + 1)) - 1ULL);
                        u64 cmB = B.comma & maskB;
                        if (__builtin_expect(__builtin_popcountll(cmB) == 2 &&
                                             (quote_faults(B.comma | B.nl, B.quote) & maskB) == 0 &&
                                             (pos + c1 + 1 + e) < end, 1)) {
                            u32 c2 = u32(__builtin_ctzll(cmB)); cmB &= cmB - 1;
                            u32 c3 = u32(__builtin_ctzll(cmB));
                            u32 o0 = u32(A.quote & 1u);
                            u32 o1 = u32(B.quote & 1u);
                            u32 o2 = u32((B.quote >> (c2 + 1)) & 1u);
                            u32 o3 = u32((B.quote >> (c3 + 1)) & 1u);
                            fl[0] = {p + o0,           c1 - 2u * o0};
                            fl[1] = {q + o1,           c2 - 2u * o1};
                            fl[2] = {q + c2 + 1 + o2,  c3 - c2 - 1u - 2u * o2};
                            fl[3] = {q + c3 + 1 + o3,  e - c3 - 1u - 2u * o3};
                            pos += size_t(c1) + 1u + size_t(e) + 1u;
                            cb(fl);
                            continue;
                        }
                    }
                }
            }
        } else if constexpr (NF == 2) {
            Mask64 A = classify(p);
            if (__builtin_expect(A.nl != 0, 1)) {
                u32 e = u32(__builtin_ctzll(A.nl));
                u64 maskA = (e == 63) ? ~0ULL : ((1ULL << (e + 1)) - 1ULL);
                u64 cmA = A.comma & maskA;
                if (__builtin_expect(__builtin_popcountll(cmA) == 1 &&
                                     (quote_faults(A.comma | A.nl, A.quote) & maskA) == 0 &&
                                     (pos + e) < end, 1)) {
                    u32 c1 = u32(__builtin_ctzll(cmA));
                    u32 o0 = u32(A.quote & 1u);
                    u32 o1 = u32((A.quote >> (c1 + 1)) & 1u);
                    fl[0] = {p + o0,          c1 - 2u * o0};
                    fl[1] = {p + c1 + 1 + o1, e - c1 - 1u - 2u * o1};
                    pos += size_t(e) + 1u;
                    cb(fl);
                    continue;
                }
            }
        }

        // Exact fallback for long lines (>2 windows, e.g., two 33-char hex IDs + quotes),
        // malformed lines, empty lines, or file tail without trailing '\n'.
        const u8* nl_ptr = static_cast<const u8*>(std::memchr(p, '\n', end - pos));
        const u8* line_end = nl_ptr ? nl_ptr : (base + end);
        pos = nl_ptr ? size_t(nl_ptr - base + 1) : end;
        if (split_exact<NF>(p, line_end, fl)) {
            cb(fl);
        }
    }
}

} // namespace hyp
