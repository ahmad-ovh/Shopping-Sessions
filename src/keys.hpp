#pragma once

#include "common.hpp"
#include <immintrin.h>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace hyp {

// ============================================================================
// Zero-Interning Lossless Dual-Width ID Encoding
//
// Domain 0 (DOM_SHORT): prefix ('u'/'e'/'k') + 1..8 non-NUL ASCII bytes (1..127).
//   - Stored verbatim in little-endian u64 `lo` (zero-padded in upper bytes).
//   - Byte-wise order between two Domain 0 keys: bswap64(a.lo) < bswap64(b.lo).
//   - Covers 87.5% of generator IDs with ZERO hashing/interning.
//
// Domain 1 (DOM_HEX32): prefix ('u'/'e'/'k') + 32 lowercase hex chars [0-9a-f].
//   - Stored losslessly in 128 bits (hi, lo) big-endian (first char = top nibble of hi).
//   - Byte-wise order between two Domain 1 keys: (a.hi, a.lo) < (b.hi, b.lo).
//   - Covers 12.5% of v5 generator IDs with ZERO interning, ZERO locks, ZERO collisions!
//
// Domain 2 (DOM_EXOTIC): any other non-empty string (uppercase prefix, spaces, \r,
//   UTF-8, other lengths). Stored in a tiny mutex-protected exact byte arena.
//   - Covers <0.001% of generator IDs and arbitrary fixture strings.
// ============================================================================

enum : u8 {
    DOM_SHORT  = 0,
    DOM_HEX32  = 1,
    DOM_EXOTIC = 2,
};

struct Hex128 {
    u64 hi;
    u64 lo;
    bool operator==(const Hex128& o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const Hex128& o) const { return hi != o.hi || lo != o.lo; }
    bool operator<(const Hex128& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
};

struct UserKey {
    u64 lo;      // DOM_SHORT: raw little-endian 1..8 suffix bytes; DOM_HEX32: lo 64 bits; DOM_EXOTIC: intern id
    u64 hi;      // DOM_HEX32: hi 64 bits; else 0
    u8  dom;     // DOM_SHORT / DOM_HEX32 / DOM_EXOTIC

    bool operator==(const UserKey& o) const {
        return dom == o.dom && lo == o.lo && hi == o.hi;
    }
    bool operator!=(const UserKey& o) const { return !(*this == o); }
};

inline u64 hash_short(u64 w) { return wyhash64(w); }
inline u64 hash_hex128(u64 hi, u64 lo) { return wyhash128(hi, lo); }
inline u64 hash_user_key(const UserKey& k) {
    if (k.dom == DOM_SHORT) return wyhash64(k.lo);
    if (k.dom == DOM_HEX32) return wyhash128(k.hi, k.lo);
    return wyhash64(k.lo ^ 0x9e3779b97f4a7c15ULL);
}

// Fast AVX2 lossless parser: 32 lowercase hex ASCII bytes -> big-endian Hex128
// Returns true iff all 32 bytes are strictly in '0'..'9' or 'a'..'f'.
static inline bool parse_hex32_avx2(const u8* p, Hex128& out) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

    // Check '0'..'9': (v - '0') <= 9 using unsigned saturating-style signed compare
    // Signed range check: x in [lo, hi] <=> (x ^ 0x80) in [lo^0x80, hi^0x80]
    // Even simpler: all valid bytes have bit 7 == 0; check directly:
    __m256i is_dig = _mm256_and_si256(
        _mm256_cmpgt_epi8(v, _mm256_set1_epi8('0' - 1)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8('9' + 1), v));
    __m256i is_af = _mm256_and_si256(
        _mm256_cmpgt_epi8(v, _mm256_set1_epi8('a' - 1)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8('f' + 1), v));
    __m256i ok = _mm256_or_si256(is_dig, is_af);
    if (__builtin_expect(_mm256_movemask_epi8(ok) != -1, 0)) {
        return false;
    }

    // Map '0'..'9' -> 0..9 (sub 48), 'a'..'f' -> 10..15 (sub 87 = 48 + 39)
    __m256i sub = _mm256_add_epi8(_mm256_set1_epi8('0'),
                                  _mm256_and_si256(is_af, _mm256_set1_epi8('a' - '0' - 10)));
    __m256i nib = _mm256_sub_epi8(v, sub); // 32 nibbles in 0..15

    // Pair adjacent nibbles: (nib[2i] << 4) | nib[2i+1] via maddubs with [16, 1, 16, 1, ...]
    __m256i weights = _mm256_set1_epi16(0x0110); // byte 0 * 16 + byte 1 * 1
    __m256i pairs16 = _mm256_maddubs_epi16(nib, weights); // 16 x u16 (each in 0..255)

    // Pack 16 x u16 across two 128-bit lanes into 16 contiguous bytes:
    __m128i lo128 = _mm256_castsi256_si128(pairs16);
    __m128i hi128 = _mm256_extracti128_si256(pairs16, 1);
    __m128i bytes16 = _mm_packus_epi16(lo128, hi128);

    // bytes16[0..7] is big-endian hi, bytes16[8..15] is big-endian lo
    u64 h_le = u64(_mm_cvtsi128_si64(bytes16));
    u64 l_le = u64(_mm_extract_epi64(bytes16, 1));
    out.hi = __builtin_bswap64(h_le);
    out.lo = __builtin_bswap64(l_le);
    return true;
}

// Fast AVX2 formatter: big-endian Hex128 -> 32 lowercase hex ASCII bytes
static inline void format_hex32_avx2(Hex128 h, u8* dst) {
    u64 h_le = __builtin_bswap64(h.hi);
    u64 l_le = __builtin_bswap64(h.lo);
    __m128i b = _mm_set_epi64x(i64(l_le), i64(h_le)); // 16 bytes in order b[0..15]

    // Split into high nibbles and low nibbles
    __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(b, 4), mask0f);
    __m128i lo_nib = _mm_and_si128(b, mask0f);

    // Interleave: [hi0, lo0, hi1, lo1, ...]
    __m128i out_lo = _mm_unpacklo_epi8(hi_nib, lo_nib); // first 16 hex chars
    __m128i out_hi = _mm_unpackhi_epi8(hi_nib, lo_nib); // last 16 hex chars

    __m128i lut = _mm_setr_epi8('0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f');
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_shuffle_epi8(lut, out_lo));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_shuffle_epi8(lut, out_hi));
}

// Unpack top 32 bits of big-endian Hex128.hi into 8 ASCII hex bytes as big-endian u64
// Used for O(1) comparison between Domain 0 (Short) and Domain 1 (Hex32)!
static inline u64 hex128_top8_be(u64 hi) {
    u32 top = u32(hi >> 32); // 8 nibbles, MSB first
    u32 t_le = __builtin_bswap32(top);
    __m128i b = _mm_cvtsi32_si128(i32(t_le));
    __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128i hi_n = _mm_and_si128(_mm_srli_epi16(b, 4), mask0f);
    __m128i lo_n = _mm_and_si128(b, mask0f);
    __m128i nibs = _mm_unpacklo_epi8(hi_n, lo_n);
    __m128i lut = _mm_setr_epi8('0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f');
    __m128i chars = _mm_shuffle_epi8(lut, nibs);
    return __builtin_bswap64(u64(_mm_cvtsi128_si64(chars)));
}

// Fast check & pack for Domain 0: prefix + 1..8 non-NUL ASCII bytes
template <u8 Prefix>
static inline bool try_pack_short(const u8* p, size_t n, u64& out_word) {
    if (__builtin_expect(n < 2 || n > 9 || p[0] != Prefix, 0)) return false;
    const size_t tlen = n - 1;
    // Safe 8-byte load: MFile guarantees >= 64 bytes padding past EOF,
    // and caller buffers are padded. Mask off bytes beyond tlen.
    u64 word;
    std::memcpy(&word, p + 1, 8);
    const u64 mask = (tlen == 8) ? ~u64(0) : ((u64(1) << (tlen * 8)) - 1);
    word &= mask;
    // Reject NUL (0) inside tlen bytes or any non-ASCII byte (>= 0x80)
    const u64 zero_bytes = (word - 0x0101010101010101ULL) & ~word & mask;
    if (__builtin_expect(((zero_bytes | word) & 0x8080808080808080ULL) != 0, 0)) return false;
    out_word = word;
    return true;
}

template <u8 Prefix>
static inline bool try_pack_hex32(const u8* p, size_t n, Hex128& out_hex) {
    if (__builtin_expect(n != 33 || p[0] != Prefix, 0)) return false;
    return parse_hex32_avx2(p + 1, out_hex);
}

// Tiny exact string interner ONLY for Domain 2 (Exotic IDs: <0.001% of data + fixtures)
class ExoticInterner {
    mutable std::mutex mu_;
    std::vector<u8> arena_;
    std::vector<u32> offs_{0}; // string i is arena_[offs_[i] .. offs_[i+1]]
    std::unordered_map<std::string, u32> map_;

public:
    ExoticInterner() {
        arena_.reserve(65536);
        offs_.reserve(4096);
    }

    u32 intern(const u8* p, size_t n) {
        std::string key(reinterpret_cast<const char*>(p), n);
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        u32 id = u32(offs_.size() - 1);
        arena_.insert(arena_.end(), p, p + n);
        offs_.push_back(u32(arena_.size()));
        map_.emplace(std::move(key), id);
        return id;
    }

    // Lock-free read after all interning for a phase is done (or call under lock)
    std::string_view get_unlocked(u32 id) const {
        u32 a = offs_[id], b = offs_[id + 1];
        return std::string_view(reinterpret_cast<const char*>(arena_.data() + a), b - a);
    }

    std::string_view get_locked(u32 id) const {
        std::lock_guard<std::mutex> lk(mu_);
        return get_unlocked(id);
    }

    size_t count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return offs_.size() - 1;
    }
};

template <u8 Prefix>
static inline size_t decode_short(u64 w, u8 buf[9]) {
    buf[0] = Prefix;
    std::memcpy(buf + 1, &w, 8);
    return 1u + (8u - (size_t(__builtin_clzll(w)) >> 3));
}

template <u8 Prefix>
static inline size_t decode_hex32(Hex128 h, u8 buf[33]) {
    buf[0] = Prefix;
    format_hex32_avx2(h, buf + 1);
    return 33;
}

// Decode any UserKey into caller's stack buffer (returns string_view pointing to buf or exotic arena)
static inline std::string_view decode_user_key(const UserKey& k, u8 buf[33], const ExoticInterner& ex) {
    if (__builtin_expect(k.dom == DOM_SHORT, 1)) {
        size_t len = decode_short<'u'>(k.lo, buf);
        return std::string_view(reinterpret_cast<const char*>(buf), len);
    }
    if (__builtin_expect(k.dom == DOM_HEX32, 1)) {
        decode_hex32<'u'>({k.hi, k.lo}, buf);
        return std::string_view(reinterpret_cast<const char*>(buf), 33);
    }
    return ex.get_unlocked(u32(k.lo));
}

// Exact byte-wise less-than comparison across ALL 3 domains!
static inline bool user_key_less(const UserKey& a, const UserKey& b, const ExoticInterner& ex) {
    if (__builtin_expect(a.dom == b.dom, 1)) {
        if (a.dom == DOM_SHORT) return __builtin_bswap64(a.lo) < __builtin_bswap64(b.lo);
        if (a.dom == DOM_HEX32) return a.hi != b.hi ? (a.hi < b.hi) : (a.lo < b.lo);
        if (a.lo == b.lo) return false;
        return ex.get_unlocked(u32(a.lo)) < ex.get_unlocked(u32(b.lo));
    }
    // Cross-domain Short vs Hex32: O(1) without full decode!
    if (a.dom == DOM_SHORT && b.dom == DOM_HEX32) {
        u64 sa = __builtin_bswap64(a.lo);
        u64 hb = hex128_top8_be(b.hi);
        return sa <= hb; // if equal on 8 bytes, len(short)<=9 < 33 => short < hex32!
    }
    if (a.dom == DOM_HEX32 && b.dom == DOM_SHORT) {
        u64 ha = hex128_top8_be(a.hi);
        u64 sb = __builtin_bswap64(b.lo);
        return ha < sb; // if equal on 8 bytes, hex32 > short!
    }
    // At least one is DOM_EXOTIC
    u8 ba[33], bb[33];
    return decode_user_key(a, ba, ex) < decode_user_key(b, bb, ex);
}

static inline UserKey encode_user_key(const u8* p, size_t n, ExoticInterner& ex) {
    u64 sw;
    if (try_pack_short<'u'>(p, n, sw)) return {sw, 0, DOM_SHORT};
    Hex128 hx;
    if (try_pack_hex32<'u'>(p, n, hx)) return {hx.lo, hx.hi, DOM_HEX32};
    return {u64(ex.intern(p, n)), 0, DOM_EXOTIC};
}

} // namespace hyp
