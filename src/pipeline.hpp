#pragma once

#include "catalog_links.hpp"
#include "common.hpp"
#include "csv_time.hpp"
#include "keys.hpp"
#include "parquet.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace HYP_NS {

// ============================================================================
// Cache-Line-Packed Records & Non-Temporal Write-Combining Staging Buffers
// ============================================================================

struct alignas(64) ShortPGroup5 {
    u64 ukey[5];
    u32 ts[5];
    u32 count; // 1..5 valid entries in this group
};
static_assert(sizeof(ShortPGroup5) == 64);

static constexpr u32 SHORT_PBLK_GROUPS = 63; // 63 * 64 + 64 = 4096 B exact!
static constexpr u32 SHORT_PBLK_CAP    = SHORT_PBLK_GROUPS * 5; // 315 events/block
struct alignas(64) ShortPBlock {
    ShortPGroup5 g[SHORT_PBLK_GROUPS]; // 0..4031: 100% pure NT-streamed!
    ShortPBlock* next;                 // 4032..4095: written only when block finishes!
    u32 ngroups;
    u32 idx;
    u32 pad[12];
};
static_assert(sizeof(ShortPBlock) == 4096);

struct ShortPStageMeta {
    ShortPGroup5* dst;
    u32 base_seq;
    u16 rem;
    u16 count;
};
static_assert(sizeof(ShortPStageMeta) == 16);

struct LongPRec {
    u64 hi;
    u64 lo;
    u32 ts;
    u32 pad;
};
static constexpr u32 LONG_PBLK_CAP = 40;
struct alignas(64) LongPBlock {
    LongPBlock* next;
    u32 n;
    u32 idx;
    u32 pad[12];
    LongPRec r[LONG_PBLK_CAP];
};

struct ExoticPRec {
    u32 eid;
    u32 ts;
};
static constexpr u32 EXOTIC_PBLK_CAP = 30;
struct alignas(64) ExoticPBlock {
    ExoticPBlock* next;
    u32 n;
    u32 idx;
    ExoticPRec r[EXOTIC_PBLK_CAP];
};

struct PurchRec {
    u64 u_lo;
    u64 u_hi;
    u32 ts;
    u32 price;
    u32 ploc;
    u16 cat;
    u8  u_dom;
    u8  pad;
};
static constexpr u32 PURCH_BLK_CAP = 30;
struct alignas(64) PurchBlock {
    PurchBlock* next;
    u32 n;
    u32 pad;
    PurchRec r[PURCH_BLK_CAP];
};

// 3 ShortClicks packed SoA into ONE 64-byte cache line (24 + 12 + 12 + 12 + 4 = 64 B)!
struct alignas(64) ShortClickGroup3 {
    u64 ukey[3];
    u32 ts[3];
    u32 cid_off[3];
    u32 camp_file_len[3]; // (camp << 18) | (file << 8) | len
    u32 count;
};
static_assert(sizeof(ShortClickGroup3) == 64);

static constexpr u32 SHORT_CKBLK_GROUPS = 63; // 63 * 64 + 64 = 4096 B exact!
struct alignas(64) ShortClickBlock {
    ShortClickGroup3 g[SHORT_CKBLK_GROUPS];
    ShortClickBlock* next;
    u32 ngroups;
    u32 pad[13];
};
static_assert(sizeof(ShortClickBlock) == 4096);

struct ShortClickStageMeta {
    ShortClickGroup3* dst;
    u32 total;
    u16 rem;
    u16 count;
};
static_assert(sizeof(ShortClickStageMeta) == 16);

// Fallback click record for DOM_HEX32 / DOM_EXOTIC / large campaign or click_id
struct OtherClickRec {
    u64 u_lo;
    u64 u_hi;
    u32 ts;
    u32 camp_dom;
    u32 file_len;
    u32 cid_off;
};
static constexpr u32 OTHER_CKBLK_CAP = 30;
struct alignas(64) OtherClickBlock {
    OtherClickBlock* next;
    u32 n;
    u32 pad;
    OtherClickRec r[OTHER_CKBLK_CAP];
};

// Exact disjoint dedup records (4 x 16B ShortERec = 1 cache line!)
struct ShortERec {
    u64 ekey;
    u32 ing_sec;
    u32 loc;
};
struct alignas(64) ShortEGroup4 {
    ShortERec r[4];
};
static constexpr u32 SHORT_EBLK_GROUPS = 63; // 63 * 64 + 64 = 4096 B exact!
struct alignas(64) ShortEBlock {
    ShortERec r[SHORT_EBLK_GROUPS * 4]; // 252 records (0..4031: 100% pure NT-streamed!)
    ShortEBlock* next;
    u32 n;
    u32 pad[13];
};
static_assert(sizeof(ShortEBlock) == 4096);

struct ShortEStageMeta {
    ShortERec* dst;
    u32 total;
    u16 rem;
    u16 count;
};
static_assert(sizeof(ShortEStageMeta) == 16);

struct LongERec {
    u64 hi;
    u64 lo;
    u32 ing_sec;
    u32 loc;
};
static constexpr u32 LONG_EBLK_CAP = 168;
struct alignas(64) LongEBlock {
    LongEBlock* next;
    u32 n;
    u32 pad;
    LongERec r[LONG_EBLK_CAP];
};

struct ExoticERec {
    u32 eid;
    u32 loc;
    i64 ing_ms;
};

static inline u32 pack_loc(u8 dom, int tid, u32 seq) {
    return (u32(dom) << 30) | (u32(tid) << 26) | seq;
}

static inline void copy64_nt(void* dst, const void* src) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(src);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst), v);
#else
    const __m256i* s = reinterpret_cast<const __m256i*>(src);
    __m256i* d = reinterpret_cast<__m256i*>(dst);
    __m256i v0 = _mm256_load_si256(s);
    __m256i v1 = _mm256_load_si256(s + 1);
    _mm256_stream_si256(d, v0);
    _mm256_stream_si256(d + 1, v1);
#endif
}

static inline void copy64_wb(void* dst, const void* src) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(src);
    _mm512_store_si512(dst, v);
#else
    const __m256i* s = reinterpret_cast<const __m256i*>(src);
    __m256i* d = reinterpret_cast<__m256i*>(dst);
    __m256i v0 = _mm256_load_si256(s);
    __m256i v1 = _mm256_load_si256(s + 1);
    _mm256_store_si256(d, v0);
    _mm256_store_si256(d + 1, v1);
#endif
}

template <class Block>
struct BlockList {
    Block* head = nullptr;
    Block* tail = nullptr;
    u32 total = 0;
};

// Hot 8-byte campaign-day counters (224 KB/thread at 2000 campaigns x 14 days -> fits in L2!)
struct CampDayHot {
    u32 sessions = 0;
    u32 assisted = 0;
};
// Cold 16-byte purchase counters (ONLY touched on ~3% purchasing sessions!)
struct CampDayPurch {
    u32 purchasing_sessions = 0;
    u32 pad = 0;
    i64 revenue_cents = 0;
};

struct alignas(64) Worker {
    int tid = 0;
    bool use_nt = true;
    SlabArena p_arena;
    SlabArena e_arena;

    std::vector<ShortPBlock*>  short_pidx;
    std::vector<LongPBlock*>   long_pidx;
    std::vector<ExoticPBlock*> exotic_pidx;

    // L1/L2-resident write-combining staging buffers + compact 16B metadata (zero block-header touches!)
    ShortPGroup5*       sp_stage  = nullptr;
    ShortPStageMeta*    sp_meta   = nullptr;
    ShortClickGroup3*   sck_stage = nullptr;
    ShortClickStageMeta* sck_meta = nullptr;
    ShortEGroup4*       se_stage  = nullptr;
    ShortEStageMeta*    se_meta   = nullptr;

    std::vector<BlockList<ShortPBlock>>     sp_parts;
    std::vector<BlockList<LongPBlock>>      lp_parts;
    std::vector<BlockList<ExoticPBlock>>    ep_parts;
    std::vector<BlockList<PurchBlock>>      pu_parts;
    std::vector<BlockList<ShortClickBlock>> sck_parts;
    std::vector<BlockList<OtherClickBlock>> ock_parts;

    std::vector<BlockList<ShortEBlock>> se_parts;
    std::vector<BlockList<LongEBlock>>  le_parts;
    std::vector<ExoticERec> exotic_erecs;

    std::vector<ShortERec> dtab_short;
    std::vector<LongERec>  dtab_long;
    std::vector<u8>        dtab_long_used;

    pq::UserDailyCols ud_cols;
    std::vector<CampDayHot>   cd_hot;
    std::vector<CampDayPurch> cd_purch;

    u32 P_ = 0, DS_ = 0;

    void init(int id, u32 P, u32 DS, u32 DL, u32 ncamp_plus1, u32 ndays, bool nt_stores, size_t arena_chunk = 48ULL << 20) {
        tid = id;
        use_nt = nt_stores;
        P_ = P;
        DS_ = DS;
        p_arena.set_chunk_size(arena_chunk);
        e_arena.set_chunk_size(arena_chunk);
        sp_parts.resize(P);
        lp_parts.resize(P);
        ep_parts.resize(P);
        pu_parts.resize(P);
        sck_parts.resize(P);
        ock_parts.resize(P);
        se_parts.resize(DS);
        le_parts.resize(DL);

        sp_stage  = static_cast<ShortPGroup5*>(p_arena.alloc(size_t(P) * sizeof(ShortPGroup5), 64));
        sp_meta   = static_cast<ShortPStageMeta*>(p_arena.alloc(size_t(P) * sizeof(ShortPStageMeta), 64));
        std::memset(sp_meta, 0, size_t(P) * sizeof(ShortPStageMeta));

        sck_stage = static_cast<ShortClickGroup3*>(p_arena.alloc(size_t(P) * sizeof(ShortClickGroup3), 64));
        sck_meta  = static_cast<ShortClickStageMeta*>(p_arena.alloc(size_t(P) * sizeof(ShortClickStageMeta), 64));
        std::memset(sck_meta, 0, size_t(P) * sizeof(ShortClickStageMeta));

        se_stage  = static_cast<ShortEGroup4*>(e_arena.alloc(size_t(DS) * sizeof(ShortEGroup4), 64));
        se_meta   = static_cast<ShortEStageMeta*>(e_arena.alloc(size_t(DS) * sizeof(ShortEStageMeta), 64));
        std::memset(se_meta, 0, size_t(DS) * sizeof(ShortEStageMeta));

        short_pidx.reserve(16384);
        long_pidx.reserve(4096);
        exotic_pidx.reserve(64);
        cd_hot.assign(size_t(ncamp_plus1) * size_t(ndays), CampDayHot{});
        cd_purch.assign(size_t(ncamp_plus1) * size_t(ndays), CampDayPurch{});
    }

    inline void copy64(void* dst, const void* src) const {
        if (use_nt) copy64_nt(dst, src);
        else        copy64_wb(dst, src);
    }

    __attribute__((noinline)) void alloc_short_p_block(u32 p) {
        auto& lst = sp_parts[p];
        if (lst.tail) lst.tail->ngroups = SHORT_PBLK_GROUPS;
        ShortPBlock* b = static_cast<ShortPBlock*>(p_arena.alloc(sizeof(ShortPBlock), 64));
        b->next = nullptr;
        b->ngroups = 0;
        b->idx = u32(short_pidx.size());
        short_pidx.push_back(b);
        if (lst.tail) lst.tail->next = b;
        else lst.head = b;
        lst.tail = b;
        lst.total += SHORT_PBLK_CAP; // upper bound; exact adjusted in flush_all_stages!
        ShortPStageMeta& m = sp_meta[p];
        m.dst = &b->g[0];
        m.base_seq = b->idx * SHORT_PBLK_CAP;
        m.rem = SHORT_PBLK_GROUPS;
    }

    inline u32 emit_short_p(u32 p, u64 ukey, u32 ts) {
        ShortPStageMeta& m = sp_meta[p];
        if (__builtin_expect(m.rem == 0, 0)) alloc_short_p_block(p);
        ShortPGroup5& st = sp_stage[p];
        u32 slot = m.count++;
        st.ukey[slot] = ukey;
        st.ts[slot] = ts;
        u32 loc = pack_loc(DOM_SHORT, tid, m.base_seq + slot);
        if (__builtin_expect(slot == 4, 0)) {
            st.count = 5;
            copy64(m.dst, &st);
            m.dst++;
            m.base_seq += 5u;
            m.rem--;
            m.count = 0;
        }
        return loc;
    }

    __attribute__((noinline)) void alloc_short_e_block(u32 q) {
        auto& lst = se_parts[q];
        if (lst.tail) lst.tail->n = SHORT_EBLK_GROUPS * 4u;
        ShortEBlock* b = static_cast<ShortEBlock*>(e_arena.alloc(sizeof(ShortEBlock), 64));
        b->next = nullptr;
        b->n = 0;
        if (lst.tail) lst.tail->next = b;
        else lst.head = b;
        lst.tail = b;
        ShortEStageMeta& m = se_meta[q];
        m.dst = &b->r[0];
        m.rem = SHORT_EBLK_GROUPS;
    }

    inline void emit_short_e(u32 q, const ShortERec& er) {
        ShortEStageMeta& m = se_meta[q];
        u32 slot = m.count++;
        se_stage[q].r[slot] = er;
        m.total++;
        if (__builtin_expect(slot == 3, 0)) {
            if (__builtin_expect(m.rem == 0, 0)) alloc_short_e_block(q);
            copy64(m.dst, &se_stage[q]);
            m.dst += 4;
            m.rem--;
            m.count = 0;
        }
    }

    __attribute__((noinline)) void alloc_short_ck_block(u32 p) {
        auto& lst = sck_parts[p];
        if (lst.tail) lst.tail->ngroups = SHORT_CKBLK_GROUPS;
        ShortClickBlock* b = static_cast<ShortClickBlock*>(p_arena.alloc(sizeof(ShortClickBlock), 64));
        b->next = nullptr;
        b->ngroups = 0;
        if (lst.tail) lst.tail->next = b;
        else lst.head = b;
        lst.tail = b;
        ShortClickStageMeta& m = sck_meta[p];
        m.dst = &b->g[0];
        m.rem = SHORT_CKBLK_GROUPS;
    }

    inline void emit_short_click(u32 p, u64 ukey, u32 ts, u32 cid_off, u32 cfl) {
        ShortClickStageMeta& m = sck_meta[p];
        ShortClickGroup3& st = sck_stage[p];
        u32 slot = m.count++;
        st.ukey[slot] = ukey;
        st.ts[slot] = ts;
        st.cid_off[slot] = cid_off;
        st.camp_file_len[slot] = cfl;
        m.total++;
        if (__builtin_expect(slot == 2, 0)) {
            if (__builtin_expect(m.rem == 0, 0)) alloc_short_ck_block(p);
            st.count = 3;
            copy64(m.dst, &st);
            m.dst++;
            m.rem--;
            m.count = 0;
        }
    }

    inline void flush_all_stages() {
        for (u32 p = 0; p < P_; ++p) {
            ShortPStageMeta& m = sp_meta[p];
            if (m.count > 0) {
                sp_stage[p].count = m.count;
                copy64(m.dst, &sp_stage[p]);
                m.rem--;
                m.count = 0;
            }
            if (sp_parts[p].tail) {
                u32 used_g = SHORT_PBLK_GROUPS - m.rem;
                sp_parts[p].tail->ngroups = used_g;
                sp_parts[p].total -= (m.rem * 5u);
            }
        }
        for (u32 p = 0; p < P_; ++p) {
            ShortClickStageMeta& m = sck_meta[p];
            if (m.count > 0) {
                if (m.rem == 0) alloc_short_ck_block(p);
                sck_stage[p].count = m.count;
                copy64(m.dst, &sck_stage[p]);
                m.rem--;
                m.count = 0;
            }
            if (sck_parts[p].tail) {
                sck_parts[p].tail->ngroups = SHORT_CKBLK_GROUPS - m.rem;
            }
            sck_parts[p].total = m.total;
        }
        for (u32 q = 0; q < DS_; ++q) {
            ShortEStageMeta& m = se_meta[q];
            if (m.count > 0) {
                if (m.rem == 0) alloc_short_e_block(q);
                std::memcpy(m.dst, &se_stage[q].r[0], size_t(m.count) * sizeof(ShortERec));
                se_parts[q].tail->n = (SHORT_EBLK_GROUPS - m.rem) * 4u + m.count;
                m.count = 0;
            } else if (se_parts[q].tail) {
                se_parts[q].tail->n = (SHORT_EBLK_GROUPS - m.rem) * 4u;
            }
            se_parts[q].total = m.total;
        }
        if (use_nt) _mm_sfence();
    }

    inline u32 emit_long_p(u32 p, u64 hi, u64 lo, u32 ts) {
        auto& lst = lp_parts[p];
        LongPBlock* b = lst.tail;
        if (__builtin_expect(!b || b->n == LONG_PBLK_CAP, 0)) {
            b = static_cast<LongPBlock*>(p_arena.alloc(sizeof(LongPBlock), 64));
            b->next = nullptr;
            b->n = 0;
            b->idx = u32(long_pidx.size());
            long_pidx.push_back(b);
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        u32 slot = b->n++;
        b->r[slot] = {hi, lo, ts, 0};
        lst.total++;
        return pack_loc(DOM_HEX32, tid, b->idx * LONG_PBLK_CAP + slot);
    }

    inline u32 emit_exotic_p(u32 p, u32 eid, u32 ts) {
        auto& lst = ep_parts[p];
        ExoticPBlock* b = lst.tail;
        if (__builtin_expect(!b || b->n == EXOTIC_PBLK_CAP, 0)) {
            b = static_cast<ExoticPBlock*>(p_arena.alloc(sizeof(ExoticPBlock), 64));
            b->next = nullptr;
            b->n = 0;
            b->idx = u32(exotic_pidx.size());
            exotic_pidx.push_back(b);
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        u32 slot = b->n++;
        b->r[slot] = {eid, ts};
        lst.total++;
        return pack_loc(DOM_EXOTIC, tid, b->idx * EXOTIC_PBLK_CAP + slot);
    }

    inline void emit_purch(u32 p, const PurchRec& pr) {
        auto& lst = pu_parts[p];
        PurchBlock* b = lst.tail;
        if (__builtin_expect(!b || b->n == PURCH_BLK_CAP, 0)) {
            b = static_cast<PurchBlock*>(p_arena.alloc(sizeof(PurchBlock), 64));
            b->next = nullptr;
            b->n = 0;
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        b->r[b->n++] = pr;
        lst.total++;
    }

    inline void emit_other_click(u32 p, const OtherClickRec& cr) {
        auto& lst = ock_parts[p];
        OtherClickBlock* b = lst.tail;
        if (__builtin_expect(!b || b->n == OTHER_CKBLK_CAP, 0)) {
            b = static_cast<OtherClickBlock*>(p_arena.alloc(sizeof(OtherClickBlock), 64));
            b->next = nullptr;
            b->n = 0;
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        b->r[b->n++] = cr;
        lst.total++;
    }

    inline void emit_long_e(u32 q, const LongERec& er) {
        auto& lst = le_parts[q];
        LongEBlock* b = lst.tail;
        if (__builtin_expect(!b || b->n == LONG_EBLK_CAP, 0)) {
            b = static_cast<LongEBlock*>(e_arena.alloc(sizeof(LongEBlock), 64));
            b->next = nullptr;
            b->n = 0;
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        b->r[b->n++] = er;
        lst.total++;
    }
};

// ============================================================================
// Global Pipeline Context
// ============================================================================

struct Pipeline {
    std::string data_dir;
    std::string out_dir;
    int ncpu = 8;

    i64 window_start = 0;
    i64 window_end = 0;
    i64 click_start = 0;
    i64 base_day = 0;
    u32 ws_mod = 0;
    u32 num_days = 1;
    i64 ing_origin_sec = 0;
    u32 num_bins = 0;

    FastTime ftime;
    Catalogs cats;
    ExoticInterner user_exotic;
    ExoticInterner event_exotic;
    LinkResolver links;

    std::vector<MFile> link_files;
    std::vector<MFile> event_files;
    std::vector<MFile> click_files;

    u32 P = 512;
    u32 DS = 1024;
    u32 DL = 256;
    bool use_nt_stores = true;

    std::vector<Worker> workers;
    Profiler prof;

    inline u32* loc_ts_ptr(u32 loc) {
        u8 dom = u8(loc >> 30);
        int tid = int((loc >> 26) & 0xFu);
        u32 seq = loc & ((1u << 26) - 1u);
        Worker& w = workers[tid];
        if (__builtin_expect(dom == DOM_SHORT, 1)) {
            u32 blk = seq / SHORT_PBLK_CAP;
            u32 rem = seq - blk * SHORT_PBLK_CAP;
            u32 grp = rem / 5u;
            u32 slt = rem - grp * 5u;
            return &w.short_pidx[blk]->g[grp].ts[slt];
        }
        if (dom == DOM_HEX32) {
            return &w.long_pidx[seq / LONG_PBLK_CAP]->r[seq % LONG_PBLK_CAP].ts;
        }
        return &w.exotic_pidx[seq / EXOTIC_PBLK_CAP]->r[seq % EXOTIC_PBLK_CAP].ts;
    }

    inline void kill_loc(u32 loc) {
        if (loc == INVALID_LOC) return;
        __atomic_store_n(loc_ts_ptr(loc), DEAD_TS, __ATOMIC_RELAXED);
    }

    inline bool is_loc_dead(u32 loc) {
        return *loc_ts_ptr(loc) == DEAD_TS;
    }

    // ------------------------------------------------------------------------
    // Phase 1: Parse Events & Clicks with 64-Row Prefetched Bloom + Alias Pipeline
    // ------------------------------------------------------------------------
    struct PendEvent {
        u64 eid_lo;
        u64 eid_hi;
        u64 u_lo;
        u64 u_hi;
        u64 uhash;
        u32 ts_rel;
        u32 ing_sec;
        u32 price;
        u16 cat;
        u8  eid_dom;
        u8  u_dom;
    };

    struct PendClick {
        UserKey raw_user;
        u64 uhash;
        u32 ts_rel;
        u32 camp;
        u32 file_len;
        u32 cid_off;
    };

    void flush_events(Worker& w, PendEvent* buf, int n) {
        if (n == 0) return;
        const bool check_links = links.has_aliases;

        if (check_links) {
            u8 hit_idx[64];
            int nhit = 0;
            for (int i = 0; i < n; ++i) {
                if (buf[i].u_dom != DOM_EXOTIC) links.prefetch_bloom(buf[i].uhash);
            }
            for (int i = 0; i < n; ++i) {
                u8 d = buf[i].u_dom;
                if (__builtin_expect(d == DOM_SHORT, 1)) {
                    if (links.bloom_may_contain(buf[i].uhash)) {
                        links.prefetch_short_alias(buf[i].uhash);
                        hit_idx[nhit++] = u8(i);
                    }
                } else if (d == DOM_HEX32) {
                    if (links.bloom_may_contain(buf[i].uhash)) {
                        links.prefetch_hex_alias(buf[i].uhash);
                        hit_idx[nhit++] = u8(i);
                    }
                } else {
                    hit_idx[nhit++] = u8(i);
                }
            }
            for (int h = 0; h < nhit; ++h) {
                PendEvent& pe = buf[hit_idx[h]];
                UserKey canon{pe.u_lo, pe.u_hi, pe.u_dom};
                if (__builtin_expect(canon.dom == DOM_SHORT, 1)) {
                    canon = links.resolve_short_hit(canon.lo, pe.uhash);
                } else if (canon.dom == DOM_HEX32) {
                    canon = links.resolve_hex_hit(canon.hi, canon.lo, pe.uhash);
                } else {
                    auto it = links.exotic_aliases.find(u32(canon.lo));
                    if (it != links.exotic_aliases.end()) canon = it->second;
                }
                if (canon.dom != pe.u_dom || canon.lo != pe.u_lo || canon.hi != pe.u_hi) {
                    pe.u_lo = canon.lo;
                    pe.u_hi = canon.hi;
                    pe.u_dom = canon.dom;
                    pe.uhash = hash_user_key(canon);
                }
            }
        }

        const u32 pmask = P - 1u;
        const u32 dsmask = DS - 1u;
        const u32 dlmask = DL - 1u;

        // Pass 4a: Straight-line P + Purchase scatter (L1-resident sp_stage, zero alias branches!)
        u32 locs[64];
        for (int i = 0; i < n; ++i) {
            const PendEvent& pe = buf[i];
            u32 p = u32(pe.uhash >> 32) & pmask;
            u32 loc;
            if (__builtin_expect(pe.u_dom == DOM_SHORT, 1)) {
                loc = w.emit_short_p(p, pe.u_lo, pe.ts_rel);
            } else if (pe.u_dom == DOM_HEX32) {
                loc = w.emit_long_p(p, pe.u_hi, pe.u_lo, pe.ts_rel);
            } else {
                loc = w.emit_exotic_p(p, u32(pe.u_lo), pe.ts_rel);
            }
            locs[i] = loc;
            if (__builtin_expect(pe.price != 0, 0)) {
                w.emit_purch(p, {pe.u_lo, pe.u_hi, pe.ts_rel, pe.price, loc, pe.cat, pe.u_dom, 0});
            }
        }

        // Pass 4b: Straight-line E dedup scatter (L1/L2-resident se_stage, zero P interference!)
        // Precompute next record's hash 1 iteration ahead to prevent SSBD store-to-load stalls!
        u32 q_nxt = (n > 0 && buf[0].eid_dom == DOM_SHORT)
                    ? (u32(wyhash64(buf[0].eid_lo) >> 32) & dsmask) : 0u;
        for (int i = 0; i < n; ++i) {
            const PendEvent& pe = buf[i];
            if (__builtin_expect(pe.eid_dom == DOM_SHORT, 1)) {
                u32 q = q_nxt;
                if (i + 1 < n && __builtin_expect(buf[i + 1].eid_dom == DOM_SHORT, 1)) {
                    q_nxt = u32(wyhash64(buf[i + 1].eid_lo) >> 32) & dsmask;
                }
                w.emit_short_e(q, {pe.eid_lo, pe.ing_sec, locs[i]});
            } else {
                u32 q = u32(wyhash128(pe.eid_hi, pe.eid_lo) >> 32) & dlmask;
                w.emit_long_e(q, {pe.eid_hi, pe.eid_lo, pe.ing_sec, locs[i]});
                if (i + 1 < n && buf[i + 1].eid_dom == DOM_SHORT) {
                    q_nxt = u32(wyhash64(buf[i + 1].eid_lo) >> 32) & dsmask;
                }
            }
        }
    }

    void flush_clicks(Worker& w, PendClick* buf, int n) {
        if (n == 0) return;
        const bool check_links = links.has_aliases;

        if (check_links) {
            u8 hit_idx[64];
            int nhit = 0;
            for (int i = 0; i < n; ++i) {
                if (buf[i].raw_user.dom != DOM_EXOTIC) links.prefetch_bloom(buf[i].uhash);
            }
            for (int i = 0; i < n; ++i) {
                u8 d = buf[i].raw_user.dom;
                if (__builtin_expect(d == DOM_SHORT, 1)) {
                    if (links.bloom_may_contain(buf[i].uhash)) {
                        links.prefetch_short_alias(buf[i].uhash);
                        hit_idx[nhit++] = u8(i);
                    }
                } else if (d == DOM_HEX32) {
                    if (links.bloom_may_contain(buf[i].uhash)) {
                        links.prefetch_hex_alias(buf[i].uhash);
                        hit_idx[nhit++] = u8(i);
                    }
                } else {
                    hit_idx[nhit++] = u8(i);
                }
            }
            for (int h = 0; h < nhit; ++h) {
                PendClick& pc = buf[hit_idx[h]];
                UserKey canon = pc.raw_user;
                if (__builtin_expect(canon.dom == DOM_SHORT, 1)) {
                    canon = links.resolve_short_hit(canon.lo, pc.uhash);
                } else if (canon.dom == DOM_HEX32) {
                    canon = links.resolve_hex_hit(canon.hi, canon.lo, pc.uhash);
                } else {
                    auto it = links.exotic_aliases.find(u32(canon.lo));
                    if (it != links.exotic_aliases.end()) canon = it->second;
                }
                if (canon != pc.raw_user) {
                    pc.raw_user = canon;
                    pc.uhash = hash_user_key(canon);
                }
            }
        }

        const u32 pmask = P - 1u;
        for (int i = 0; i < n; ++i) {
            const PendClick& pc = buf[i];
            const UserKey& canon = pc.raw_user;
            u32 p = u32(pc.uhash >> 32) & pmask;
            u32 fi = pc.file_len >> 20;
            u32 clen = pc.file_len & 0xFFFFFu;
            if (__builtin_expect(canon.dom == DOM_SHORT && pc.camp < 16384u && fi < 1024u && clen < 256u, 1)) {
                u32 cfl = (pc.camp << 18) | (fi << 8) | clen;
                w.emit_short_click(p, canon.lo, pc.ts_rel, pc.cid_off, cfl);
            } else {
                u32 camp_dom = (u32(canon.dom) << 30) | pc.camp;
                w.emit_other_click(p, {canon.lo, canon.hi, pc.ts_rel, camp_dom, pc.file_len, pc.cid_off});
            }
        }
    }

    void parse_events_and_clicks(ThreadPool& pool) {
        struct Task { u8 kind; u32 file; size_t s, e; };
        std::vector<Task> tasks;
        tasks.reserve(4096);

        constexpr size_t EV_CSZ = 2u << 20;
        for (u32 fi = 0; fi < event_files.size(); ++fi) {
            const MFile& mf = event_files[fi];
            if (mf.size == 0) continue;
            const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
            if (!nl) continue;
            size_t body = size_t(nl - mf.data) + 1;
            if (body >= mf.size) continue;
            for (size_t off = body; off < mf.size; off += EV_CSZ) {
                tasks.push_back({0, fi, off, std::min(off + EV_CSZ, mf.size)});
            }
        }
        constexpr size_t CK_CSZ = 2u << 20;
        for (u32 fi = 0; fi < click_files.size(); ++fi) {
            const MFile& mf = click_files[fi];
            if (mf.size == 0) continue;
            const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
            if (!nl) continue;
            size_t body = size_t(nl - mf.data) + 1;
            if (body >= mf.size) continue;
            for (size_t off = body; off < mf.size; off += CK_CSZ) {
                tasks.push_back({1, fi, off, std::min(off + CK_CSZ, mf.size)});
            }
        }

        std::atomic<size_t> next_task{0};
        double ev_core_ms[16]{}, ck_core_ms[16]{};
        pool.run([&](int tid) {
            Worker& w = workers[tid];
            PendEvent ev_buf[64];
            PendClick ck_buf[64];
            int nev = 0, nck = 0;
            double my_ev = 0.0, my_ck = 0.0;

            auto handle_event_line = [&](const Field fl[7]) {
                if (__builtin_expect(fl[0].n == 0, 0)) return;

                i64 ing_sec_abs = ftime.parse_ingest_sec(fl[3].p, fl[3].n);
                i64 ing_ms;
                bool ing_compact;
                i64 ing_diff = 0;
                if (__builtin_expect(ing_sec_abs != INVALID_TS, 1)) {
                    ing_ms = ing_sec_abs * 1000LL;
                    ing_diff = ing_sec_abs - ing_origin_sec;
                    ing_compact = (ing_diff >= 0 && ing_diff < 0xFFFFFFFELL);
                } else {
                    ing_ms = parse_ts_general(fl[3].p, fl[3].n);
                    if (ing_ms == INVALID_TS) return;
                    ing_diff = (ing_ms / 1000LL) - ing_origin_sec;
                    ing_compact = (ing_diff >= 0 && ing_diff < 0xFFFFFFFELL && (ing_ms % 1000LL) == 0);
                }

                bool valid = (fl[1].n > 0 && fl[5].n > 0);
                i64 ts = INVALID_TS;
                if (__builtin_expect(valid, 1)) {
                    ts = ftime.parse(fl[2].p, fl[2].n);
                    valid = (ts >= window_start && ts < window_end && (ing_ms - ts) <= DAY_MS);
                }

                u64 eid_lo = 0, eid_hi = 0;
                u8 eid_dom = DOM_EXOTIC;
                if (__builtin_expect(ing_compact, 1)) {
                    u64 sw;
                    if (try_pack_short<'e'>(fl[0].p, fl[0].n, sw)) {
                        eid_lo = sw;
                        eid_dom = DOM_SHORT;
                    } else {
                        Hex128 hx;
                        if (try_pack_hex32<'e'>(fl[0].p, fl[0].n, hx)) {
                            eid_hi = hx.hi;
                            eid_lo = hx.lo;
                            eid_dom = DOM_HEX32;
                        }
                    }
                }
                if (__builtin_expect(eid_dom == DOM_EXOTIC, 0)) {
                    eid_lo = event_exotic.intern(fl[0].p, fl[0].n);
                }

                u32 price = 0;
                u16 cat = 0;
                if (__builtin_expect(valid, 1)) {
                    const u8* tp = fl[4].p;
                    u32 tn = fl[4].n;
                    if (tn == 4) {
                        u32 w4;
                        std::memcpy(&w4, tp, 4);
                        valid = (w4 == 0x77656976u /*"view"*/) || (w4 == 0x74726163u /*"cart"*/);
                    } else if (tn == 8) {
                        u64 w8;
                        std::memcpy(&w8, tp, 8);
                        if (w8 == 0x6573616863727570ULL /*"purchase"*/) {
                            valid = parse_price_cents(fl[6].p, fl[6].n, price);
                            if (valid) cat = cats.lookup_product(fl[5].p, fl[5].n);
                        } else {
                            valid = false;
                        }
                    } else {
                        valid = false;
                    }
                }

                if (__builtin_expect(!valid, 0)) {
                    if (eid_dom == DOM_SHORT) {
                        u32 q = u32(wyhash64(eid_lo) >> 32) & (DS - 1u);
                        w.emit_short_e(q, {eid_lo, u32(ing_diff), INVALID_LOC});
                    } else if (eid_dom == DOM_HEX32) {
                        u32 q = u32(wyhash128(eid_hi, eid_lo) >> 32) & (DL - 1u);
                        w.emit_long_e(q, {eid_hi, eid_lo, u32(ing_diff), INVALID_LOC});
                    } else {
                        w.exotic_erecs.push_back({u32(eid_lo), INVALID_LOC, ing_ms});
                    }
                    return;
                }

                UserKey uk = encode_user_key(fl[1].p, fl[1].n, user_exotic);
                if (__builtin_expect(eid_dom == DOM_EXOTIC, 0)) {
                    UserKey canon = uk;
                    if (links.has_aliases) {
                        u64 uh = hash_user_key(uk);
                        if (canon.dom == DOM_SHORT) {
                            if (links.bloom_may_contain(uh)) canon = links.resolve_short_hit(canon.lo, uh);
                        } else if (canon.dom == DOM_HEX32) {
                            if (links.bloom_may_contain(uh)) canon = links.resolve_hex_hit(canon.hi, canon.lo, uh);
                        } else {
                            auto it = links.exotic_aliases.find(u32(canon.lo));
                            if (it != links.exotic_aliases.end()) canon = it->second;
                        }
                    }
                    u32 p = u32(hash_user_key(canon) >> 32) & (P - 1u);
                    u32 ts_rel = u32(ts - window_start);
                    u32 loc = (canon.dom == DOM_SHORT) ? w.emit_short_p(p, canon.lo, ts_rel)
                            : (canon.dom == DOM_HEX32) ? w.emit_long_p(p, canon.hi, canon.lo, ts_rel)
                            : w.emit_exotic_p(p, u32(canon.lo), ts_rel);
                    if (price != 0) w.emit_purch(p, {canon.lo, canon.hi, ts_rel, price, loc, cat, canon.dom, 0});
                    w.exotic_erecs.push_back({u32(eid_lo), loc, ing_ms});
                    return;
                }

                PendEvent& pe = ev_buf[nev++];
                pe.eid_lo = eid_lo;
                pe.eid_hi = eid_hi;
                pe.u_lo = uk.lo;
                pe.u_hi = uk.hi;
                pe.uhash = hash_user_key(uk);
                pe.ts_rel = u32(ts - window_start);
                pe.ing_sec = u32(ing_diff);
                pe.price = price;
                pe.cat = cat;
                pe.eid_dom = eid_dom;
                pe.u_dom = uk.dom;
                if (nev == 64) {
                    flush_events(w, ev_buf, nev);
                    nev = 0;
                }
            };

            for (;;) {
                size_t ti = next_task.fetch_add(1, std::memory_order_relaxed);
                if (ti >= tasks.size()) break;
                const Task& tk = tasks[ti];

                if (tk.kind == 0) {
                    double t0 = prof.enabled ? now_sec() : 0.0;
                    const MFile& mf = event_files[tk.file];
                    size_t s = tk.s;
                    if (s > 0 && mf.data[s - 1] != '\n') {
                        const u8* nl = static_cast<const u8*>(std::memchr(mf.data + s, '\n', mf.size - s));
                        if (!nl) continue;
                        s = size_t(nl - mf.data) + 1;
                    }
                    if (s >= tk.e) continue;
                    size_t e = tk.e;
                    if (e < mf.size && mf.data[e - 1] != '\n') {
                        const u8* nl = static_cast<const u8*>(std::memchr(mf.data + e, '\n', mf.size - e));
                        e = nl ? size_t(nl - mf.data) + 1 : mf.size;
                    }
                    scan_csv<7>(mf.data, s, e, handle_event_line);
                    if (prof.enabled) my_ev += (now_sec() - t0) * 1000.0;
                } else {
                    double t0 = prof.enabled ? now_sec() : 0.0;
                    const MFile& mf = click_files[tk.file];
                    size_t s = tk.s;
                    if (s > 0 && mf.data[s - 1] != '\n') {
                        const u8* nl = static_cast<const u8*>(std::memchr(mf.data + s, '\n', mf.size - s));
                        if (!nl) continue;
                        s = size_t(nl - mf.data) + 1;
                    }
                    if (s >= tk.e) continue;
                    size_t e = tk.e;
                    if (e < mf.size && mf.data[e - 1] != '\n') {
                        const u8* nl = static_cast<const u8*>(std::memchr(mf.data + e, '\n', mf.size - e));
                        e = nl ? size_t(nl - mf.data) + 1 : mf.size;
                    }
                    const u32 fshift = tk.file << 20;
                    scan_csv<4>(mf.data, s, e, [&](const Field fl[4]) {
                        if (__builtin_expect(fl[0].n == 0 || fl[1].n == 0, 0)) return;
                        i32 camp = cats.lookup_campaign(fl[2].p, fl[2].n);
                        if (__builtin_expect(camp < 0, 0)) return;
                        i64 ts = ftime.parse(fl[3].p, fl[3].n);
                        if (__builtin_expect(ts < click_start || ts >= window_end, 0)) return;

                        UserKey uk = encode_user_key(fl[1].p, fl[1].n, user_exotic);
                        PendClick& pc = ck_buf[nck++];
                        pc.raw_user = uk;
                        pc.uhash = hash_user_key(uk);
                        pc.ts_rel = u32(ts - click_start);
                        pc.camp = u32(camp);
                        pc.file_len = fshift | (fl[0].n & 0xFFFFFu);
                        pc.cid_off = u32(fl[0].p - mf.data);
                        if (nck == 64) {
                            flush_clicks(w, ck_buf, nck);
                            nck = 0;
                        }
                    });
                    if (prof.enabled) my_ck += (now_sec() - t0) * 1000.0;
                }
            }
            flush_events(w, ev_buf, nev);
            flush_clicks(w, ck_buf, nck);
            w.flush_all_stages();
            if (prof.enabled) { ev_core_ms[tid] = my_ev; ck_core_ms[tid] = my_ck; }
        });
        if (prof.enabled) {
            double se = 0, sc = 0;
            for (int t = 0; t < pool.size(); ++t) { se += ev_core_ms[t]; sc += ck_core_ms[t]; }
            prof.note("parse:events_avg", se / pool.size());
            prof.note("parse:clicks_avg", sc / pool.size());
        }
    }

    // ------------------------------------------------------------------------
    // Phase 2: Parallel L2-Resident Exact Deduplication (kills ~3% losers in place!)
    // ------------------------------------------------------------------------
    void dedup_phase(ThreadPool& pool) {
        const int nt = pool.size();
        const u32 total_tasks = DS + DL;
        std::atomic<u32> next_q{0};

        pool.run([&](int tid) {
            Worker& w = workers[tid];
            for (;;) {
                u32 task = next_q.fetch_add(1, std::memory_order_relaxed);
                if (task >= total_tasks) break;

                if (task < DS) {
                    u32 q = task;
                    u32 n = 0;
                    for (int t = 0; t < nt; ++t) n += workers[t].se_parts[q].total;
                    if (n == 0) continue;

                    u32 cap = 16;
                    while (cap < n * 2) cap <<= 1;
                    if (w.dtab_short.size() < cap) w.dtab_short.resize(cap);
                    ShortERec* tab = w.dtab_short.data();
                    std::memset(tab, 0, size_t(cap) * sizeof(ShortERec));
                    const u32 mask = cap - 1u;

                    for (int t = 0; t < nt; ++t) {
                        for (ShortEBlock* b = workers[t].se_parts[q].head; b; b = b->next) {
                            const u32 bn = b->n;
                            if (bn == 0) continue;
                            const ShortERec* rr = b->r;
                            u32 nxt_idx = u32(wyhash64(rr[0].ekey)) & mask;
                            for (u32 i = 0; i < bn; ++i) {
                                const ShortERec e = rr[i];
                                u32 idx = nxt_idx;
                                if (i + 1 < bn) nxt_idx = u32(wyhash64(rr[i + 1].ekey)) & mask;
                                for (;;) {
                                    ShortERec& s = tab[idx];
                                    if (s.ekey == 0) {
                                        s = e;
                                        break;
                                    }
                                    if (s.ekey == e.ekey) {
                                        if (e.ing_sec < s.ing_sec) {
                                            kill_loc(s.loc);
                                            s = e;
                                        } else {
                                            kill_loc(e.loc);
                                        }
                                        break;
                                    }
                                    idx = (idx + 1u) & mask;
                                }
                            }
                        }
                    }
                } else {
                    u32 q = task - DS;
                    u32 n = 0;
                    for (int t = 0; t < nt; ++t) n += workers[t].le_parts[q].total;
                    if (n == 0) continue;

                    u32 cap = 16;
                    while (cap < n * 2) cap <<= 1;
                    if (w.dtab_long.size() < cap) w.dtab_long.resize(cap);
                    if (w.dtab_long_used.size() < cap) w.dtab_long_used.resize(cap);
                    LongERec* tab = w.dtab_long.data();
                    u8* used = w.dtab_long_used.data();
                    std::memset(used, 0, cap);
                    const u32 mask = cap - 1u;

                    for (int t = 0; t < nt; ++t) {
                        for (LongEBlock* b = workers[t].le_parts[q].head; b; b = b->next) {
                            const u32 bn = b->n;
                            for (u32 i = 0; i < bn; ++i) {
                                const LongERec& e = b->r[i];
                                u32 idx = u32(wyhash128(e.hi, e.lo)) & mask;
                                for (;;) {
                                    if (!used[idx]) {
                                        used[idx] = 1;
                                        tab[idx] = e;
                                        break;
                                    }
                                    LongERec& s = tab[idx];
                                    if (s.hi == e.hi && s.lo == e.lo) {
                                        if (e.ing_sec < s.ing_sec) {
                                            kill_loc(s.loc);
                                            s = e;
                                        } else {
                                            kill_loc(e.loc);
                                        }
                                        break;
                                    }
                                    idx = (idx + 1u) & mask;
                                }
                            }
                        }
                    }
                }
            }
        });

        size_t total_ex = 0;
        for (int t = 0; t < nt; ++t) total_ex += workers[t].exotic_erecs.size();
        if (total_ex > 0) {
            std::unordered_map<u32, std::pair<i64, u32>> emap;
            emap.reserve(total_ex * 2);
            for (int t = 0; t < nt; ++t) {
                for (const auto& e : workers[t].exotic_erecs) {
                    auto it = emap.find(e.eid);
                    if (it == emap.end()) {
                        emap.emplace(e.eid, std::make_pair(e.ing_ms, e.loc));
                    } else {
                        if (e.ing_ms < it->second.first) {
                            kill_loc(it->second.second);
                            it->second = {e.ing_ms, e.loc};
                        } else {
                            kill_loc(e.loc);
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Phase 3: Prefetched L1 Grouping + Small-User / Bitmask Sessionization +
    //          O(S+C) Sliding-Window Attribution & Assists + Zero-Copy writev Parquet
    // ------------------------------------------------------------------------
    struct LocalClick {
        u32 ts;
        u32 camp;
        u32 file_len;
        u32 cid_off;
    };

    struct LocalPurch {
        u32 ts;
        u32 price;
        u16 cat;
    };

    struct Bin {
        u32 mn;
        u32 mx;
        u32 cnt;
        u32 sid;
    };

    struct Session {
        u32 start;
        u32 nev;
        u64 rev;
        u16 day;
        u8  has_p;
        u8  pad;
    };

    void sessions_and_write(ThreadPool& pool) {
        const int nt = pool.size();
        const u32 ncamp = cats.num_campaigns;
        const u32 ncats = u32(cats.categories.size());
        const bool use_binning = (num_bins > 0 && num_bins <= 1024);
        const u32 nbin_words = (num_bins + 63u) >> 6;
        const u32 wsm = ws_mod;

        // LPT scheduling: sort partitions descending by total events so whale partitions start FIRST!
        std::vector<std::pair<u32, u32>> p_order(P);
        u64 grand_ev = 0;
        for (u32 p = 0; p < P; ++p) {
            u32 tot = 0;
            for (int t = 0; t < nt; ++t) {
                tot += workers[t].sp_parts[p].total + workers[t].lp_parts[p].total + workers[t].ep_parts[p].total;
            }
            p_order[p] = {tot, p};
            grand_ev += tot;
        }
        std::sort(p_order.begin(), p_order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        const size_t est_rows_thr = std::max<size_t>(size_t(grand_ev / u64(nt * 6u)) + 16384u, 32768u);
        const size_t est_ppl_thr  = std::max<size_t>(est_rows_thr / 3u + 4096u, 8192u);

        std::atomic<u32> next_pi{0};
        double grp_ms[16]{}, per_ms[16]{}, pq_ms[16]{};

        pool.run([&](int tid) {
            Worker& w = workers[tid];
            w.e_arena.rewind();
            w.ud_cols.reserve(est_rows_thr, est_ppl_thr, &w.e_arena);

            struct ShortLSlot {
                u64 key;
                u32 pid;
                u32 pad;
            };
            struct OtherLSlot {
                UserKey key;
                u32 pid;
                u8 used;
            };
            std::vector<ShortLSlot> stab;
            std::vector<OtherLSlot> otab;
            std::vector<UserKey> lkeys;
            std::vector<u32> ev_cnt, pu_cnt, ck_cnt;
            std::vector<u32> ev_off, pu_off, ck_off;

            std::vector<u32> tmp_pid, tmp_ts;
            std::vector<u32> ev_ts;
            std::vector<u32> tmp_pu_pid;
            std::vector<LocalPurch> tmp_pu, pu_arr;
            std::vector<u32> tmp_ck_pid;
            std::vector<LocalClick> tmp_ck, ck_arr;

            std::vector<Bin> bins(use_binning ? num_bins : 1, Bin{0xFFFFFFFFu, 0, 0, 0});
            std::vector<u64> bmap(use_binning ? nbin_words : 1, 0);

            // Flat L1-resident Session array (zero push_back checks!)
            std::vector<Session> sess_buf(use_binning ? (num_bins + 4u) : 1024u);
            std::vector<u64> catday(size_t(num_days) * size_t(ncats), 0);

            std::vector<u32> ccnt(ncamp, 0);
            std::vector<u32> cact(ncamp, 0);

            CampDayHot*   cdh = w.cd_hot.data();
            CampDayPurch* cdp = w.cd_purch.data();
            CampDayHot*   org_h = cdh + size_t(ncamp) * size_t(num_days);
            CampDayPurch* org_p = cdp + size_t(ncamp) * size_t(num_days);

            auto click_id_sv = [&](const LocalClick& c) -> std::string_view {
                u32 fi = c.file_len >> 20;
                u32 len = c.file_len & 0xFFFFFu;
                return std::string_view(
                    reinterpret_cast<const char*>(click_files[fi].data + c.cid_off), len);
            };

            double my_grp = 0.0, my_per = 0.0;

            for (;;) {
                u32 pi = next_pi.fetch_add(1, std::memory_order_relaxed);
                if (pi >= P) break;
                if (p_order[pi].first == 0) break;
                const u32 p = p_order[pi].second;

                double t_g0 = prof.enabled ? now_sec() : 0.0;

                u32 est_sp = 0, est_lp = 0, est_ep = 0, est_pu = 0, est_ck = 0;
                for (int t = 0; t < nt; ++t) {
                    est_sp += workers[t].sp_parts[p].total;
                    est_lp += workers[t].lp_parts[p].total;
                    est_ep += workers[t].ep_parts[p].total;
                    est_pu += workers[t].pu_parts[p].total;
                    est_ck += workers[t].sck_parts[p].total + workers[t].ock_parts[p].total;
                }
                u32 est_ev = est_sp + est_lp + est_ep;
                if (est_ev == 0) continue;

                // Cap initial stab/otab so whale partitions (e.g. 400K events from 1 whale user + 1.3K normal users)
                // stay in L1/L2 cache (64 KB) instead of allocating & memset-ing a 2 MB table!
                // grow_stab()/grow_otab() automatically double if distinct persons exceed half capacity.
                u32 scap = 128;
                u32 swant = std::min<u32>(std::max<u32>(est_sp / 8u + 64u, 64u), 4096u);
                while (scap < swant) scap <<= 1;
                if (stab.size() < scap) stab.resize(scap);
                std::memset(stab.data(), 0, size_t(scap) * sizeof(ShortLSlot));
                u32 smask = scap - 1u;
                u32 s_used = 0;

                u32 ocap = 64;
                u32 owant = std::min<u32>(std::max<u32>((est_lp + est_ep) / 8u + 32u, 32u), 1024u);
                while (ocap < owant) ocap <<= 1;
                if (otab.size() < ocap) otab.resize(ocap);
                for (u32 i = 0; i < ocap; ++i) otab[i].used = 0;
                u32 omask = ocap - 1u;
                u32 o_used = 0;

                lkeys.clear();
                ev_cnt.clear();
                if (tmp_pid.size() < est_ev) {
                    tmp_pid.resize(est_ev);
                    tmp_ts.resize(est_ev);
                }
                u32* tp_out = tmp_pid.data();
                u32* tt_out = tmp_ts.data();
                u32 n_live_ev = 0;

                auto grow_stab = [&]() {
                    u32 ncap = scap << 1;
                    std::vector<ShortLSlot> ntab(ncap, ShortLSlot{0, 0, 0});
                    u32 nmask = ncap - 1u;
                    for (u32 i = 0; i < scap; ++i) {
                        if (stab[i].key == 0) continue;
                        u32 idx = u32(hash_short(stab[i].key)) & nmask;
                        while (ntab[idx].key != 0) idx = (idx + 1u) & nmask;
                        ntab[idx] = stab[i];
                    }
                    stab.swap(ntab);
                    scap = ncap;
                    smask = nmask;
                };

                auto grow_otab = [&]() {
                    u32 ncap = ocap << 1;
                    std::vector<OtherLSlot> ntab(ncap);
                    for (u32 i = 0; i < ncap; ++i) ntab[i].used = 0;
                    u32 nmask = ncap - 1u;
                    for (u32 i = 0; i < ocap; ++i) {
                        if (!otab[i].used) continue;
                        u32 idx = u32(hash_user_key(otab[i].key)) & nmask;
                        while (ntab[idx].used) idx = (idx + 1u) & nmask;
                        ntab[idx] = otab[i];
                    }
                    otab.swap(ntab);
                    ocap = ncap;
                    omask = nmask;
                };

                auto get_or_add_short_h = [&](u64 key, u32 h32) -> u32 {
                    if (__builtin_expect(s_used * 2 >= scap, 0)) grow_stab();
                    ShortLSlot* stp = stab.data();
                    u32 idx = h32 & smask;
                    for (;;) {
                        ShortLSlot& s = stp[idx];
                        if (s.key == key) return s.pid;
                        if (s.key == 0) {
                            u32 pid = u32(lkeys.size());
                            s.key = key;
                            s.pid = pid;
                            s_used++;
                            lkeys.push_back({key, 0, DOM_SHORT});
                            ev_cnt.push_back(0);
                            return pid;
                        }
                        idx = (idx + 1u) & smask;
                    }
                };

                auto get_or_add_other = [&](const UserKey& k) -> u32 {
                    if (__builtin_expect(o_used * 2 >= ocap, 0)) grow_otab();
                    u32 idx = u32(hash_user_key(k)) & omask;
                    for (;;) {
                        OtherLSlot& s = otab[idx];
                        if (!s.used) {
                            u32 pid = u32(lkeys.size());
                            s.key = k;
                            s.pid = pid;
                            s.used = 1;
                            o_used++;
                            lkeys.push_back(k);
                            ev_cnt.push_back(0);
                            return pid;
                        }
                        if (s.key == k) return s.pid;
                        idx = (idx + 1u) & omask;
                    }
                };

                auto find_short_h = [&](u64 key, u32 h32) -> i32 {
                    const ShortLSlot* stp = stab.data();
                    u32 idx = h32 & smask;
                    for (;;) {
                        const ShortLSlot& s = stp[idx];
                        if (s.key == 0) return -1;
                        if (s.key == key) return i32(s.pid);
                        idx = (idx + 1u) & smask;
                    }
                };

                auto find_any = [&](u64 lo, u64 hi, u8 dom) -> i32 {
                    if (__builtin_expect(dom == DOM_SHORT, 1)) return find_short_h(lo, u32(hash_short(lo)));
                    UserKey k{lo, hi, dom};
                    u32 idx = u32(hash_user_key(k)) & omask;
                    for (;;) {
                        const OtherLSlot& s = otab[idx];
                        if (!s.used) return -1;
                        if (s.key == k) return i32(s.pid);
                        idx = (idx + 1u) & omask;
                    }
                };

                // SINGLE HASH PASS over surviving events with 1-group-ahead prefetch (zero double-hash!)
                for (int t = 0; t < nt; ++t) {
                    for (ShortPBlock* b = workers[t].sp_parts[p].head; b; b = b->next) {
                        const u32 ng = b->ngroups;
                        if (ng == 0) continue;
                        u32 hcur[5], hnxt[5];
                        {
                            const ShortPGroup5& g0 = b->g[0];
                            ShortLSlot* stp = stab.data();
                            for (u32 k = 0; k < g0.count; ++k) {
                                hcur[k] = u32(hash_short(g0.ukey[k]));
                                __builtin_prefetch(&stp[hcur[k] & smask], 1, 3);
                            }
                        }
                        for (u32 g = 0; g < ng; ++g) {
                            if (g + 1 < ng) {
                                const ShortPGroup5& nxt = b->g[g + 1];
                                ShortLSlot* stp = stab.data();
                                u32 nc = nxt.count;
                                for (u32 k = 0; k < nc; ++k) {
                                    hnxt[k] = u32(hash_short(nxt.ukey[k]));
                                    __builtin_prefetch(&stp[hnxt[k] & smask], 1, 3);
                                }
                            }
                            const ShortPGroup5& gr = b->g[g];
                            const u32 gc = gr.count;
                            for (u32 i = 0; i < gc; ++i) {
                                u32 ts = gr.ts[i];
                                if (__builtin_expect(ts == DEAD_TS, 0)) continue;
                                u32 pid = get_or_add_short_h(gr.ukey[i], hcur[i]);
                                ev_cnt[pid]++;
                                tp_out[n_live_ev] = pid;
                                tt_out[n_live_ev] = ts;
                                n_live_ev++;
                            }
                            for (u32 k = 0; k < 5; ++k) hcur[k] = hnxt[k];
                        }
                    }
                    for (LongPBlock* b = workers[t].lp_parts[p].head; b; b = b->next) {
                        const u32 bn = b->n;
                        for (u32 i = 0; i < bn; ++i) {
                            u32 ts = b->r[i].ts;
                            if (__builtin_expect(ts == DEAD_TS, 0)) continue;
                            u32 pid = get_or_add_other({b->r[i].lo, b->r[i].hi, DOM_HEX32});
                            ev_cnt[pid]++;
                            tp_out[n_live_ev] = pid;
                            tt_out[n_live_ev] = ts;
                            n_live_ev++;
                        }
                    }
                    for (ExoticPBlock* b = workers[t].ep_parts[p].head; b; b = b->next) {
                        const u32 bn = b->n;
                        for (u32 i = 0; i < bn; ++i) {
                            u32 ts = b->r[i].ts;
                            if (__builtin_expect(ts == DEAD_TS, 0)) continue;
                            u32 pid = get_or_add_other({b->r[i].eid, 0, DOM_EXOTIC});
                            ev_cnt[pid]++;
                            tp_out[n_live_ev] = pid;
                            tt_out[n_live_ev] = ts;
                            n_live_ev++;
                        }
                    }
                }
                const u32 np = u32(lkeys.size());
                if (np == 0) continue;

                ev_off.resize(np + 1);
                {
                    u32 run = 0;
                    for (u32 i = 0; i < np; ++i) { ev_off[i] = run; run += ev_cnt[i]; }
                    ev_off[np] = run;
                }
                if (ev_ts.size() < n_live_ev) ev_ts.resize(n_live_ev);
                std::memcpy(ev_cnt.data(), ev_off.data(), np * sizeof(u32));
                u32* ec_ptr = ev_cnt.data();
                u32* et_ptr = ev_ts.data();
                u32 i_ev = 0;
                for (; i_ev + 4 <= n_live_ev; i_ev += 4) {
                    u32 p0 = tp_out[i_ev],     t0 = tt_out[i_ev];
                    u32 p1 = tp_out[i_ev + 1], t1 = tt_out[i_ev + 1];
                    u32 p2 = tp_out[i_ev + 2], t2 = tt_out[i_ev + 2];
                    u32 p3 = tp_out[i_ev + 3], t3 = tt_out[i_ev + 3];
                    u32 d0 = ec_ptr[p0]++; et_ptr[d0] = t0;
                    u32 d1 = ec_ptr[p1]++; et_ptr[d1] = t1;
                    u32 d2 = ec_ptr[p2]++; et_ptr[d2] = t2;
                    u32 d3 = ec_ptr[p3]++; et_ptr[d3] = t3;
                }
                for (; i_ev < n_live_ev; ++i_ev) {
                    et_ptr[ec_ptr[tp_out[i_ev]]++] = tt_out[i_ev];
                }

                // Purchases
                pu_cnt.assign(np, 0);
                tmp_pu_pid.clear();
                tmp_pu.clear();
                if (est_pu > 0) {
                    for (int t = 0; t < nt; ++t) {
                        for (PurchBlock* b = workers[t].pu_parts[p].head; b; b = b->next) {
                            const u32 bn = b->n;
                            for (u32 i = 0; i < bn; ++i) {
                                const PurchRec& pr = b->r[i];
                                if (__builtin_expect(is_loc_dead(pr.ploc), 0)) continue;
                                i32 pid = find_any(pr.u_lo, pr.u_hi, pr.u_dom);
                                if (pid >= 0) {
                                    pu_cnt[pid]++;
                                    tmp_pu_pid.push_back(u32(pid));
                                    tmp_pu.push_back({pr.ts, pr.price, pr.cat});
                                }
                            }
                        }
                    }
                }
                pu_off.resize(np + 1);
                {
                    u32 run = 0;
                    for (u32 i = 0; i < np; ++i) { pu_off[i] = run; run += pu_cnt[i]; }
                    pu_off[np] = run;
                }
                const u32 total_live_pu = u32(tmp_pu.size());
                if (total_live_pu > 0) {
                    if (pu_arr.size() < total_live_pu) pu_arr.resize(total_live_pu);
                    std::memcpy(pu_cnt.data(), pu_off.data(), np * sizeof(u32));
                    for (u32 i = 0; i < total_live_pu; ++i) {
                        pu_arr[pu_cnt[tmp_pu_pid[i]]++] = tmp_pu[i];
                    }
                }

                // Clicks (prefetched ShortClickBlock + OtherClickBlock)
                ck_cnt.assign(np, 0);
                tmp_ck_pid.clear();
                tmp_ck.clear();
                if (est_ck > 0) {
                    const ShortLSlot* stp = stab.data();
                    for (int t = 0; t < nt; ++t) {
                        for (ShortClickBlock* b = workers[t].sck_parts[p].head; b; b = b->next) {
                            const u32 ng = b->ngroups;
                            if (ng == 0) continue;
                            u32 hcur[3], hnxt[3];
                            {
                                const ShortClickGroup3& g0 = b->g[0];
                                for (u32 k = 0; k < g0.count; ++k) {
                                    hcur[k] = u32(hash_short(g0.ukey[k]));
                                    __builtin_prefetch(&stp[hcur[k] & smask], 0, 3);
                                }
                            }
                            for (u32 g = 0; g < ng; ++g) {
                                if (g + 1 < ng) {
                                    const ShortClickGroup3& nxt = b->g[g + 1];
                                    for (u32 k = 0; k < nxt.count; ++k) {
                                        hnxt[k] = u32(hash_short(nxt.ukey[k]));
                                        __builtin_prefetch(&stp[hnxt[k] & smask], 0, 3);
                                    }
                                }
                                const ShortClickGroup3& gr = b->g[g];
                                const u32 gc = gr.count;
                                for (u32 i = 0; i < gc; ++i) {
                                    i32 pid = find_short_h(gr.ukey[i], hcur[i]);
                                    if (pid >= 0) {
                                        ck_cnt[pid]++;
                                        tmp_ck_pid.push_back(u32(pid));
                                        u32 cfl = gr.camp_file_len[i];
                                        u32 camp = cfl >> 18;
                                        u32 file_len = (((cfl >> 8) & 0x3FFu) << 20) | (cfl & 0xFFu);
                                        tmp_ck.push_back({gr.ts[i], camp, file_len, gr.cid_off[i]});
                                    }
                                }
                                hcur[0] = hnxt[0]; hcur[1] = hnxt[1]; hcur[2] = hnxt[2];
                            }
                        }
                        for (OtherClickBlock* b = workers[t].ock_parts[p].head; b; b = b->next) {
                            const u32 bn = b->n;
                            for (u32 i = 0; i < bn; ++i) {
                                const OtherClickRec& cr = b->r[i];
                                i32 pid = find_any(cr.u_lo, cr.u_hi, u8(cr.camp_dom >> 30));
                                if (pid >= 0) {
                                    ck_cnt[pid]++;
                                    tmp_ck_pid.push_back(u32(pid));
                                    tmp_ck.push_back({cr.ts, cr.camp_dom & 0x3FFFFFFFu, cr.file_len, cr.cid_off});
                                }
                            }
                        }
                    }
                }
                ck_off.resize(np + 1);
                {
                    u32 run = 0;
                    for (u32 i = 0; i < np; ++i) { ck_off[i] = run; run += ck_cnt[i]; }
                    ck_off[np] = run;
                }
                const u32 total_live_ck = u32(tmp_ck.size());
                if (total_live_ck > 0) {
                    if (ck_arr.size() < total_live_ck) ck_arr.resize(total_live_ck);
                    std::memcpy(ck_cnt.data(), ck_off.data(), np * sizeof(u32));
                    for (u32 i = 0; i < total_live_ck; ++i) {
                        ck_arr[ck_cnt[tmp_ck_pid[i]]++] = tmp_ck[i];
                    }
                }

                double t_p0 = prof.enabled ? now_sec() : 0.0;
                if (prof.enabled) my_grp += (t_p0 - t_g0) * 1000.0;

                // Process each person using flat Session array (ZERO push_back checks!)
                for (u32 pid = 0; pid < np; ++pid) {
                    u32 e0 = ev_off[pid], e1 = ev_off[pid + 1];
                    u32 ne = e1 - e0;
                    u32 u0 = pu_off[pid], u1 = pu_off[pid + 1];
                    u32 nu = u1 - u0;
                    u32 c0 = ck_off[pid], c1 = ck_off[pid + 1];
                    u32 nc = c1 - c0;

                    if (!use_binning && sess_buf.size() < ne) sess_buf.resize(ne);
                    Session* S = sess_buf.data();

                    auto add_assist_s = [&](u32 c, u32 s_lo, u32 s_hi) {
                        CampDayHot* row = cdh + size_t(c) * size_t(num_days);
                        for (u32 j = s_lo; j < s_hi;) {
                            u16 d = S[j].day;
                            u32 e = j + 1;
                            while (e < s_hi && S[e].day == d) ++e;
                            row[d].assisted += (e - j);
                            j = e;
                        }
                    };

                    if (ne == 1) {
                        u32 ts0 = ev_ts[e0];
                        u16 d0 = use_binning ? u16((ts0 + wsm) / u32(DAY_MS))
                                             : u16(floor_div_day(window_start + i64(ts0)) - base_day);
                        u64 rev0 = 0;
                        u8 hp0 = 0;
                        i32 best_cat = -1;
                        if (nu == 1) {
                            rev0 = pu_arr[u0].price;
                            hp0 = 1;
                            best_cat = i32(pu_arr[u0].cat);
                        } else if (nu > 1) {
                            hp0 = 1;
                            u64* cs = catday.data() + size_t(d0) * size_t(ncats);
                            for (u32 i = u0; i < u1; ++i) {
                                rev0 += pu_arr[i].price;
                                cs[pu_arr[i].cat] += pu_arr[i].price;
                            }
                            u64 bv = 0;
                            for (u32 c = 0; c < ncats; ++c) {
                                if (cs[c] > bv) { bv = cs[c]; best_cat = i32(c); }
                                cs[c] = 0;
                            }
                        }
                        w.ud_cols.add_row(i32(base_day + d0), 1, 1, i64(rev0), best_cat);
                        u8 name_buf[33];
                        std::string_view sv = decode_user_key(lkeys[pid], name_buf, user_exotic);
                        w.ud_cols.add_person_name(reinterpret_cast<const u8*>(sv.data()), u32(sv.size()), 1);
                        if (nc == 0) {
                            org_h[d0].sessions++;
                            if (__builtin_expect(hp0 != 0, 0)) {
                                org_p[d0].purchasing_sessions++;
                                org_p[d0].revenue_cents += i64(rev0);
                            }
                            continue;
                        }
                        S[0] = {ts0, 1, rev0, d0, hp0, 0};
                        u32 ns = 1;
                        LocalClick* ck = ck_arr.data() + c0;
                        if (nc <= 24) {
                            for (u32 a = 1; a < nc; ++a) {
                                LocalClick v = ck[a];
                                u32 b = a;
                                while (b > 0 && ck[b - 1].ts > v.ts) { ck[b] = ck[b - 1]; --b; }
                                ck[b] = v;
                            }
                        } else {
                            std::sort(ck, ck + nc, [](const LocalClick& a, const LocalClick& b) {
                                return a.ts < b.ts;
                            });
                        }
                        for (u32 a = 0; a + 1 < nc;) {
                            u32 b = a + 1;
                            bool mixed = false;
                            while (b < nc && ck[b].ts == ck[a].ts) {
                                mixed |= (ck[b].camp != ck[a].camp);
                                ++b;
                            }
                            if (__builtin_expect(mixed, 0)) {
                                u32 best = a;
                                std::string_view best_id = click_id_sv(ck[a]);
                                for (u32 x = a + 1; x < b; ++x) {
                                    std::string_view cid = click_id_sv(ck[x]);
                                    if (cid < best_id) { best_id = cid; best = x; }
                                }
                                if (best != b - 1) std::swap(ck[best], ck[b - 1]);
                            }
                            a = b;
                        }
                        u32 lo = 0, hi = 0;
                        u32 st = ts0 + u32(WEEK_MS);
                        while (hi < nc && ck[hi].ts <= st) {
                            u32 c = ck[hi].camp;
                            if (ccnt[c]++ == 0) cact[c] = 0;
                            hi++;
                        }
                        while (lo < hi && ck[lo].ts + u32(WEEK_MS) < st) {
                            u32 c = ck[lo].camp;
                            if (--ccnt[c] == 0) add_assist_s(c, cact[c], 0);
                            lo++;
                        }
                        u32 cr = (hi > lo) ? ck[hi - 1].camp : ncamp;
                        size_t cdi = size_t(cr) * size_t(num_days) + d0;
                        cdh[cdi].sessions++;
                        if (__builtin_expect(hp0 != 0, 0)) {
                            cdp[cdi].purchasing_sessions++;
                            cdp[cdi].revenue_cents += i64(rev0);
                        }
                        while (lo < hi) {
                            u32 c = ck[lo].camp;
                            if (--ccnt[c] == 0) add_assist_s(c, cact[c], ns);
                            lo++;
                        }
                        continue;
                    }

                    u32 ns = 0;
                    if (ne <= 8) {
                        u32 sv[8];
                        const u32* src = ev_ts.data() + e0;
                        sv[0] = src[0];
                        for (u32 i = 1; i < ne; ++i) {
                            u32 v = src[i];
                            u32 j = i;
                            while (j > 0 && sv[j - 1] > v) { sv[j] = sv[j - 1]; --j; }
                            sv[j] = v;
                        }
                        u32 prev = 0;
                        for (u32 i = 0; i < ne; ++i) {
                            u32 v = sv[i];
                            if (ns == 0 || (v - prev) > u32(SESSION_GAP_MS)) {
                                u16 dy = use_binning ? u16((v + wsm) / u32(DAY_MS))
                                                     : u16(floor_div_day(window_start + i64(v)) - base_day);
                                S[ns++] = {v, 0, 0, dy, 0, 0};
                            }
                            S[ns - 1].nev++;
                            prev = v;
                        }
                        if (nu > 0) {
                            const LocalPurch* pu = pu_arr.data() + u0;
                            for (u32 i = 0; i < nu; ++i) {
                                u32 pts = pu[i].ts;
                                u32 sid = ns - 1;
                                while (sid > 0 && S[sid].start > pts) --sid;
                                S[sid].rev += pu[i].price;
                                S[sid].has_p = 1;
                                catday[size_t(S[sid].day) * size_t(ncats) + pu[i].cat] += pu[i].price;
                            }
                        }
                    } else if (__builtin_expect(use_binning, 1)) {
                        const u32* ts = ev_ts.data() + e0;
                        Bin* B = bins.data();
                        u64* BM = bmap.data();
                        for (u32 i = 0; i < ne; ++i) {
                            u32 v = ts[i];
                            u32 b = v / u32(SESSION_GAP_MS);
                            Bin& e = B[b];
                            if (v < e.mn) e.mn = v;
                            if (v > e.mx) e.mx = v;
                            e.cnt++;
                            BM[b >> 6] |= (1ULL << (b & 63u));
                        }

                        u32 prev_mx = 0;
                        if (nu == 0) {
                            for (u32 wi = 0; wi < nbin_words; ++wi) {
                                u64 m = BM[wi];
                                if (!m) continue;
                                BM[wi] = 0;
                                while (m) {
                                    u32 b = (wi << 6) + u32(__builtin_ctzll(m));
                                    m &= m - 1ULL;
                                    Bin& e = B[b];
                                    if (ns == 0 || (e.mn - prev_mx) > u32(SESSION_GAP_MS)) {
                                        S[ns++] = {e.mn, 0, 0, u16((e.mn + wsm) / u32(DAY_MS)), 0, 0};
                                    }
                                    S[ns - 1].nev += e.cnt;
                                    prev_mx = e.mx;
                                    e = {0xFFFFFFFFu, 0, 0, 0};
                                }
                            }
                        } else {
                            for (u32 wi = 0; wi < nbin_words; ++wi) {
                                u64 m = BM[wi];
                                while (m) {
                                    u32 b = (wi << 6) + u32(__builtin_ctzll(m));
                                    m &= m - 1ULL;
                                    Bin& e = B[b];
                                    if (ns == 0 || (e.mn - prev_mx) > u32(SESSION_GAP_MS)) {
                                        S[ns++] = {e.mn, 0, 0, u16((e.mn + wsm) / u32(DAY_MS)), 0, 0};
                                    }
                                    S[ns - 1].nev += e.cnt;
                                    prev_mx = e.mx;
                                    e.sid = ns - 1;
                                }
                            }
                            const LocalPurch* pu = pu_arr.data() + u0;
                            for (u32 i = 0; i < nu; ++i) {
                                u32 sid = B[pu[i].ts / u32(SESSION_GAP_MS)].sid;
                                S[sid].rev += pu[i].price;
                                S[sid].has_p = 1;
                                catday[size_t(S[sid].day) * size_t(ncats) + pu[i].cat] += pu[i].price;
                            }
                            for (u32 wi = 0; wi < nbin_words; ++wi) {
                                u64 m = BM[wi];
                                if (!m) continue;
                                BM[wi] = 0;
                                while (m) {
                                    u32 b = (wi << 6) + u32(__builtin_ctzll(m));
                                    m &= m - 1ULL;
                                    B[b] = {0xFFFFFFFFu, 0, 0, 0};
                                }
                            }
                        }
                    } else {
                        u32* ts = ev_ts.data() + e0;
                        std::sort(ts, ts + ne);
                        u32 prev = 0;
                        for (u32 i = 0; i < ne; ++i) {
                            u32 v = ts[i];
                            if (ns == 0 || (v - prev) > u32(SESSION_GAP_MS)) {
                                S[ns++] = {v, 0, 0, u16(floor_div_day(window_start + i64(v)) - base_day), 0, 0};
                            }
                            S[ns - 1].nev++;
                            prev = v;
                        }
                        if (nu > 0) {
                            const LocalPurch* pu = pu_arr.data() + u0;
                            for (u32 i = 0; i < nu; ++i) {
                                u32 lo_s = 0, hi_s = ns;
                                while (lo_s < hi_s) {
                                    u32 mid = (lo_s + hi_s) >> 1;
                                    if (S[mid].start <= pu[i].ts) lo_s = mid + 1;
                                    else hi_s = mid;
                                }
                                u32 sid = lo_s - 1;
                                S[sid].rev += pu[i].price;
                                S[sid].has_p = 1;
                                catday[size_t(S[sid].day) * size_t(ncats) + pu[i].cat] += pu[i].price;
                            }
                        }
                    }

                    u32 active_days = 0;
                    for (u32 i = 0; i < ns;) {
                        u16 d = S[i].day;
                        u32 d_sess = 0, d_ev = 0, d_ps = 0;
                        i64 d_rev = 0;
                        do {
                            d_sess++;
                            d_ev += S[i].nev;
                            d_rev += i64(S[i].rev);
                            d_ps += S[i].has_p;
                            i++;
                        } while (i < ns && S[i].day == d);

                        i32 best_cat = -1;
                        if (d_ps > 0) {
                            u64* cs = catday.data() + size_t(d) * size_t(ncats);
                            u64 bv = 0;
                            for (u32 c = 0; c < ncats; ++c) {
                                if (cs[c] > bv) { bv = cs[c]; best_cat = i32(c); }
                                cs[c] = 0;
                            }
                        }
                        w.ud_cols.add_row(i32(base_day + d), i32(d_sess), i32(d_ev), d_rev, best_cat);
                        active_days++;

                        if (nc == 0) {
                            org_h[d].sessions += d_sess;
                            if (__builtin_expect(d_ps != 0, 0)) {
                                org_p[d].purchasing_sessions += d_ps;
                                org_p[d].revenue_cents += d_rev;
                            }
                        }
                    }

                    u8 name_buf[33];
                    std::string_view sv = decode_user_key(lkeys[pid], name_buf, user_exotic);
                    w.ud_cols.add_person_name(reinterpret_cast<const u8*>(sv.data()), u32(sv.size()), active_days);

                    if (nc == 0) continue;

                    LocalClick* ck = ck_arr.data() + c0;
                    if (nc <= 24) {
                        for (u32 a = 1; a < nc; ++a) {
                            LocalClick v = ck[a];
                            u32 b = a;
                            while (b > 0 && ck[b - 1].ts > v.ts) { ck[b] = ck[b - 1]; --b; }
                            ck[b] = v;
                        }
                    } else {
                        std::sort(ck, ck + nc, [](const LocalClick& a, const LocalClick& b) {
                            return a.ts < b.ts;
                        });
                    }

                    for (u32 a = 0; a + 1 < nc;) {
                        u32 b = a + 1;
                        bool mixed = false;
                        while (b < nc && ck[b].ts == ck[a].ts) {
                            mixed |= (ck[b].camp != ck[a].camp);
                            ++b;
                        }
                        if (__builtin_expect(mixed, 0)) {
                            u32 best = a;
                            std::string_view best_id = click_id_sv(ck[a]);
                            for (u32 x = a + 1; x < b; ++x) {
                                std::string_view cid = click_id_sv(ck[x]);
                                if (cid < best_id) { best_id = cid; best = x; }
                            }
                            if (best != b - 1) std::swap(ck[best], ck[b - 1]);
                        }
                        a = b;
                    }

                    u32 lo = 0, hi = 0;
                    for (u32 s = 0; s < ns; ++s) {
                        u32 st = S[s].start + u32(WEEK_MS);
                        while (hi < nc && ck[hi].ts <= st) {
                            u32 c = ck[hi].camp;
                            if (ccnt[c]++ == 0) cact[c] = s;
                            hi++;
                        }
                        while (lo < hi && ck[lo].ts + u32(WEEK_MS) < st) {
                            u32 c = ck[lo].camp;
                            if (--ccnt[c] == 0) add_assist_s(c, cact[c], s);
                            lo++;
                        }
                        u32 cr = (hi > lo) ? ck[hi - 1].camp : ncamp;
                        size_t cdi = size_t(cr) * size_t(num_days) + S[s].day;
                        cdh[cdi].sessions++;
                        if (__builtin_expect(S[s].has_p != 0, 0)) {
                            cdp[cdi].purchasing_sessions++;
                            cdp[cdi].revenue_cents += i64(S[s].rev);
                        }
                    }
                    while (lo < hi) {
                        u32 c = ck[lo].camp;
                        if (--ccnt[c] == 0) add_assist_s(c, cact[c], ns);
                        lo++;
                    }
                }
                if (prof.enabled) my_per += (now_sec() - t_p0) * 1000.0;
            }

            if (prof.enabled) { grp_ms[tid] = my_grp; per_ms[tid] = my_per; }

            if (w.ud_cols.nrows > 0) {
                double tpq = prof.enabled ? now_sec() : 0.0;
                std::string path = out_dir + "/user_daily/part-" + std::to_string(tid) + ".parquet";
                w.ud_cols.write_parquet(path, cats.categories);
                if (prof.enabled) pq_ms[tid] = (now_sec() - tpq) * 1000.0;
            }
        });

        if (prof.enabled) {
            double sg = 0.0, sp = 0.0, mx = 0.0;
            for (int t = 0; t < nt; ++t) { sg += grp_ms[t]; sp += per_ms[t]; if (pq_ms[t] > mx) mx = pq_ms[t]; }
            prof.note("sess:group_avg", sg / nt);
            prof.note("sess:person_avg", sp / nt);
            prof.note("user_pq:max_thread", mx);
        }

        bool any_ud = false;
        for (int t = 0; t < nt; ++t) {
            if (workers[t].ud_cols.nrows > 0) { any_ud = true; break; }
        }
        if (!any_ud) {
            workers[0].ud_cols.write_parquet(out_dir + "/user_daily/part-0.parquet", cats.categories);
        }
    }

    void write_campaign_daily() {
        const u32 ncamp_plus1 = cats.num_campaigns + 1u;
        const size_t total = size_t(ncamp_plus1) * size_t(num_days);
        std::vector<CampDayHot>   mh(total);
        std::vector<CampDayPurch> mp(total);
        for (int t = 0; t < int(workers.size()); ++t) {
            const auto& sh = workers[t].cd_hot;
            const auto& sp = workers[t].cd_purch;
            for (size_t i = 0; i < total; ++i) {
                mh[i].sessions            += sh[i].sessions;
                mh[i].assisted            += sh[i].assisted;
                mp[i].purchasing_sessions += sp[i].purchasing_sessions;
                mp[i].revenue_cents       += sp[i].revenue_cents;
            }
        }

        std::vector<pq::CampaignDailyRow> rows;
        rows.reserve(total / 2 + 16);
        for (u32 c = 0; c < ncamp_plus1; ++c) {
            for (u32 d = 0; d < num_days; ++d) {
                size_t idx = size_t(c) * size_t(num_days) + d;
                if (mh[idx].sessions > 0 || mh[idx].assisted > 0) {
                    rows.push_back({
                        c,
                        i32(base_day + d),
                        i32(mh[idx].sessions),
                        i32(mp[idx].purchasing_sessions),
                        mp[idx].revenue_cents,
                        i32(mh[idx].assisted),
                    });
                }
            }
        }

        pq::write_campaign_daily_parquet(
            out_dir + "/campaign_daily/part-0.parquet",
            rows, cats.camp_ids, cats.camp_channels);
    }

    void run(int argc, char** argv) {
        prof.init();

        const char* env_data = std::getenv("DATA_DIR");
        const char* env_out  = std::getenv("OUTPUT_DIR");
        const char* env_ncpu = std::getenv("NCPU");
        data_dir = (argc >= 3) ? argv[argc - 2] : (env_data ? env_data : "/data");
        out_dir  = (argc >= 3) ? argv[argc - 1] : (env_out  ? env_out  : "/output");
        ncpu = env_ncpu ? std::atoi(env_ncpu) : int(std::thread::hardware_concurrency());
        if (ncpu < 1) ncpu = 1;
        if (ncpu > 16) ncpu = 16;

        use_nt_stores = false;
        if (const char* ent = std::getenv("HYPERION_NT")) {
            use_nt_stores = (ent[0] != '0');
        }

        ::mkdir(out_dir.c_str(), 0777);
        ::mkdir((out_dir + "/user_daily").c_str(), 0777);
        ::mkdir((out_dir + "/campaign_daily").c_str(), 0777);

        load_manifest(data_dir + "/manifest.json", window_start, window_end);
        click_start = window_start - WEEK_MS;
        base_day = floor_div_day(window_start);
        ws_mod = u32(window_start - base_day * DAY_MS);
        i64 end_day = (window_end > window_start) ? floor_div_day(window_end - 1) : base_day;
        num_days = u32(std::max<i64>(1, end_day - base_day + 1));
        ing_origin_sec = window_start / 1000LL - 30LL * 86400LL;
        i64 win_span = std::max<i64>(0, window_end - window_start);
        num_bins = u32((win_span + SESSION_GAP_MS - 1) / SESSION_GAP_MS);
        ftime.init(window_start, window_end);

        auto lpaths = list_csv_files(data_dir + "/links");
        auto epaths = list_csv_files(data_dir + "/events");
        auto cpaths = list_csv_files(data_dir + "/clicks");

        size_t total_ev_bytes = 0;
        link_files.resize(lpaths.size());
        for (size_t i = 0; i < lpaths.size(); ++i) {
            if (!link_files[i].open_map(lpaths[i])) die("open link file");
            link_files[i].close_fd();
        }
        event_files.resize(epaths.size());
        for (size_t i = 0; i < epaths.size(); ++i) {
            if (!event_files[i].open_map(epaths[i])) die("open event file");
            event_files[i].close_fd();
            total_ev_bytes += event_files[i].size;
        }
        click_files.resize(cpaths.size());
        for (size_t i = 0; i < cpaths.size(); ++i) {
            if (!click_files[i].open_map(cpaths[i])) die("open click file");
            click_files[i].close_fd();
        }

        if (total_ev_bytes < (16u << 20)) {
            P = 64; DS = 64; DL = 32;
        } else if (total_ev_bytes < (256u << 20)) {
            P = 256; DS = 256; DL = 64;
        } else if (total_ev_bytes < (3000ULL << 20)) {
            P = 512; DS = 1024; DL = 256;
        } else {
            P = 1024; DS = 2048; DL = 512;
        }
        if (const char* ep = std::getenv("HYPERION_P"))  { u32 v = u32(std::atoi(ep)); if (v >= 16 && (v & (v - 1)) == 0) P = v; }
        if (const char* ed = std::getenv("HYPERION_DS")) { u32 v = u32(std::atoi(ed)); if (v >= 16 && (v & (v - 1)) == 0) DS = v; }

        int eff_ncpu = (total_ev_bytes < (256u << 10)) ? 1 : ncpu;
        ThreadPool pool(eff_ncpu);
        prof.mark("open+init");

        links.build_with_catalogs(link_files, cats, data_dir, pool, user_exotic, prof);
        prof.mark("cat+links+bloom");

        size_t arena_chunk = (total_ev_bytes < (256u << 20)) ? (8ULL << 20)
                           : (total_ev_bytes < (3000ULL << 20)) ? (48ULL << 20)
                           : (128ULL << 20);
        workers.resize(eff_ncpu);
        for (int t = 0; t < eff_ncpu; ++t) {
            workers[t].init(t, P, DS, DL, cats.num_campaigns + 1u, num_days, use_nt_stores, arena_chunk);
        }

        parse_events_and_clicks(pool);
        prof.mark("parse+scatter");

        dedup_phase(pool);
        prof.mark("dedup");

        sessions_and_write(pool);
        prof.mark("sessions+user_pq");

        write_campaign_daily();
        prof.mark("campaign_pq");

        // Parallel 64 MiB-chunked MADV_DONTNEED of BOTH file mappings AND anonymous arenas!
        // Eliminates single-threaded kernel exit_mmap() teardown of 500K+ PTEs after _exit(0)!
        if (eff_ncpu > 1 && total_ev_bytes > (64u << 20)) {
            std::vector<AnonRegion> chunks;
            chunks.reserve(256);
            constexpr size_t CH = 64u << 20;
            auto add_reg = [&](const void* p, size_t n) {
                if (!p || n == 0) return;
                size_t rn = (n + 4095u) & ~size_t(4095u);
                const u8* base = static_cast<const u8*>(p);
                for (size_t o = 0; o < rn; o += CH) {
                    chunks.push_back({const_cast<u8*>(base + o), std::min(CH, rn - o)});
                }
            };
            for (const auto& mf : event_files) add_reg(mf.data, mf.mapped);
            for (const auto& mf : click_files) add_reg(mf.data, mf.mapped);
            for (const auto& mf : link_files)  add_reg(mf.data, mf.mapped);
            u32 nreg = std::min<u32>(g_regions.count.load(std::memory_order_relaxed), 4096u);
            for (u32 i = 0; i < nreg; ++i) add_reg(g_regions.regs[i].ptr, g_regions.regs[i].len);

            std::atomic<size_t> ri{0};
            pool.run([&](int) {
                for (;;) {
                    size_t i = ri.fetch_add(1, std::memory_order_relaxed);
                    if (i >= chunks.size()) break;
                    ::madvise(chunks[i].ptr, chunks[i].len, MADV_DONTNEED);
                }
            });
            prof.mark("madv_release");
        }

        pool.detach_all();
        _exit(0);
    }
};

} // namespace HYP_NS
