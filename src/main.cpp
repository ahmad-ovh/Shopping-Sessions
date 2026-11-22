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

namespace hyp {

// ============================================================================
// Cache-Line-Packed Records & Write-Combining Staging Buffers
// ============================================================================

// 5 ShortP events packed into ONE 64-byte cache line (40 + 20 + 4 = 64 B)!
// Both ukey[i] and ts[i] share the EXACT same cache line (halves cache misses!).
struct alignas(64) ShortPGroup5 {
    u64 ukey[5];
    u32 ts[5];
    u32 count; // 1..5 valid entries in this group
};
static_assert(sizeof(ShortPGroup5) == 64);

static constexpr u32 SHORT_PBLK_GROUPS = 63; // 63 * 64 + 64 = 4096 B exact!
static constexpr u32 SHORT_PBLK_CAP    = SHORT_PBLK_GROUPS * 5; // 315 events/block
struct alignas(64) ShortPBlock {
    ShortPBlock* next;
    u32 ngroups;
    u32 idx;
    u32 pad[12];
    ShortPGroup5 g[SHORT_PBLK_GROUPS];
};
static_assert(sizeof(ShortPBlock) == 4096);

// Long-hex canonical user event block (12.5% of events; 1 KB block minimizes trailing waste!)
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

// Exotic canonical user event block (<0.01% of events + fixtures)
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

// Purchase side-stream (only ~3% of valid events! 1 KB blocks!)
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

// Valid click record (no global click dedup needed; lazy click_id for same-ms ties!)
struct ClickRec {
    u64 u_lo;
    u64 u_hi;
    u32 ts;
    u32 camp_dom;
    u32 file_len;
    u32 cid_off;
};
static constexpr u32 CLICK_BLK_CAP = 30;
struct alignas(64) ClickBlock {
    ClickBlock* next;
    u32 n;
    u32 pad;
    ClickRec r[CLICK_BLK_CAP];
};

// Exact disjoint dedup records (4 x 16B ShortERec = 1 cache line!)
struct ShortERec {
    u64 ekey;    // DOM_SHORT exact 64-bit raw key (always non-zero)
    u32 ing_sec; // ingest_ts_sec - ing_origin_sec
    u32 loc;     // back-pointer to PRec (or INVALID_LOC)
};
struct alignas(64) ShortEGroup4 {
    ShortERec r[4];
};
static constexpr u32 SHORT_EBLK_GROUPS = 63; // 63 * 64 + 64 = 4096 B exact!
struct alignas(64) ShortEBlock {
    ShortEBlock* next;
    u32 n; // total ShortERecs stored in this block
    u32 pad[13];
    ShortERec r[SHORT_EBLK_GROUPS * 4]; // 252 records
};
static_assert(sizeof(ShortEBlock) == 4096);

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

static inline void copy64(void* dst, const void* src) {
    const __m256i* s = reinterpret_cast<const __m256i*>(src);
    __m256i* d = reinterpret_cast<__m256i*>(dst);
    __m256i v0 = _mm256_load_si256(s);
    __m256i v1 = _mm256_load_si256(s + 1);
    _mm256_store_si256(d, v0);
    _mm256_store_si256(d + 1, v1);
}

template <class Block>
struct BlockList {
    Block* head = nullptr;
    Block* tail = nullptr;
    u32 total = 0;
};

struct CampDayStat {
    u32 sessions = 0;
    u32 purchasing_sessions = 0;
    u32 assisted_sessions = 0;
    u32 pad = 0;
    i64 revenue_cents = 0;
};

struct alignas(64) Worker {
    int tid = 0;
    SlabArena p_arena;
    SlabArena e_arena;

    std::vector<ShortPBlock*>  short_pidx;
    std::vector<LongPBlock*>   long_pidx;
    std::vector<ExoticPBlock*> exotic_pidx;

    // L1-resident write-combining staging buffers for ShortP (P * 64 B = 32 KB!)
    ShortPGroup5* sp_stage = nullptr;
    // L2-resident write-combining staging buffers for ShortE (DS * 64 B = 64 KB!)
    ShortEGroup4* se_stage = nullptr;
    std::vector<u8> se_stage_n;

    std::vector<BlockList<ShortPBlock>>  sp_parts;
    std::vector<BlockList<LongPBlock>>   lp_parts;
    std::vector<BlockList<ExoticPBlock>> ep_parts;
    std::vector<BlockList<PurchBlock>>   pu_parts;
    std::vector<BlockList<ClickBlock>>   ck_parts;

    std::vector<BlockList<ShortEBlock>> se_parts;
    std::vector<BlockList<LongEBlock>>  le_parts;
    std::vector<ExoticERec> exotic_erecs;

    std::vector<ShortERec> dtab_short;
    std::vector<LongERec>  dtab_long;
    std::vector<u8>        dtab_long_used;

    pq::UserDailyCols ud_cols;
    std::vector<CampDayStat> cd_stats;

    u32 P_ = 0, DS_ = 0;

    void init(int id, u32 P, u32 DS, u32 DL, u32 ncamp_plus1, u32 ndays) {
        tid = id;
        P_ = P;
        DS_ = DS;
        sp_parts.resize(P);
        lp_parts.resize(P);
        ep_parts.resize(P);
        pu_parts.resize(P);
        ck_parts.resize(P);
        se_parts.resize(DS);
        le_parts.resize(DL);
        sp_stage = static_cast<ShortPGroup5*>(p_arena.alloc(size_t(P) * sizeof(ShortPGroup5), 64));
        for (u32 i = 0; i < P; ++i) sp_stage[i].count = 0;
        se_stage = static_cast<ShortEGroup4*>(e_arena.alloc(size_t(DS) * sizeof(ShortEGroup4), 64));
        se_stage_n.assign(DS, 0);
        short_pidx.reserve(16384);
        long_pidx.reserve(4096);
        exotic_pidx.reserve(64);
        cd_stats.assign(size_t(ncamp_plus1) * size_t(ndays), CampDayStat{});
    }

    inline ShortPBlock* ensure_sp_blk(u32 p) {
        auto& lst = sp_parts[p];
        ShortPBlock* b = lst.tail;
        if (__builtin_expect(!b || b->ngroups == SHORT_PBLK_GROUPS, 0)) {
            b = static_cast<ShortPBlock*>(p_arena.alloc(sizeof(ShortPBlock), 64));
            b->next = nullptr;
            b->ngroups = 0;
            b->idx = u32(short_pidx.size());
            short_pidx.push_back(b);
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        return b;
    }

    inline u32 emit_short_p(u32 p, u64 ukey, u32 ts) {
        ShortPBlock* b = ensure_sp_blk(p);
        ShortPGroup5& st = sp_stage[p];
        u32 slot = st.count++;
        st.ukey[slot] = ukey;
        st.ts[slot] = ts;
        sp_parts[p].total++;
        u32 loc = pack_loc(DOM_SHORT, tid, (b->idx * SHORT_PBLK_GROUPS + b->ngroups) * 5u + slot);
        if (__builtin_expect(slot == 4, 0)) {
            copy64(&b->g[b->ngroups++], &st);
            st.count = 0;
        }
        return loc;
    }

    inline void flush_all_stages() {
        for (u32 p = 0; p < P_; ++p) {
            if (sp_stage[p].count > 0) {
                ShortPBlock* b = ensure_sp_blk(p);
                copy64(&b->g[b->ngroups++], &sp_stage[p]);
                sp_stage[p].count = 0;
            }
        }
        for (u32 q = 0; q < DS_; ++q) {
            u8 cnt = se_stage_n[q];
            if (cnt > 0) {
                auto& lst = se_parts[q];
                ShortEBlock* b = lst.tail;
                if (!b || b->n + cnt > SHORT_EBLK_GROUPS * 4u) {
                    b = static_cast<ShortEBlock*>(e_arena.alloc(sizeof(ShortEBlock), 64));
                    b->next = nullptr;
                    b->n = 0;
                    if (lst.tail) lst.tail->next = b;
                    else lst.head = b;
                    lst.tail = b;
                }
                std::memcpy(&b->r[b->n], &se_stage[q].r[0], size_t(cnt) * sizeof(ShortERec));
                b->n += cnt;
                se_stage_n[q] = 0;
            }
        }
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

    inline void emit_click(u32 p, const ClickRec& cr) {
        auto& lst = ck_parts[p];
        ClickBlock* b = lst.tail;
        if (__builtin_expect(!b || b->n == CLICK_BLK_CAP, 0)) {
            b = static_cast<ClickBlock*>(p_arena.alloc(sizeof(ClickBlock), 64));
            b->next = nullptr;
            b->n = 0;
            if (lst.tail) lst.tail->next = b;
            else lst.head = b;
            lst.tail = b;
        }
        b->r[b->n++] = cr;
        lst.total++;
    }

    inline void emit_short_e(u32 q, const ShortERec& er) {
        u8 slot = se_stage_n[q]++;
        se_stage[q].r[slot] = er;
        se_parts[q].total++;
        if (__builtin_expect(slot == 3, 0)) {
            auto& lst = se_parts[q];
            ShortEBlock* b = lst.tail;
            if (__builtin_expect(!b || b->n == SHORT_EBLK_GROUPS * 4u, 0)) {
                b = static_cast<ShortEBlock*>(e_arena.alloc(sizeof(ShortEBlock), 64));
                b->next = nullptr;
                b->n = 0;
                if (lst.tail) lst.tail->next = b;
                else lst.head = b;
                lst.tail = b;
            }
            copy64(&b->r[b->n], &se_stage[q]);
            b->n += 4u;
            se_stage_n[q] = 0;
        }
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
        u32 price;   // 0 if not purchase
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
        u8 hit[64];

        if (check_links) {
            for (int i = 0; i < n; ++i) {
                if (buf[i].u_dom != DOM_EXOTIC) links.prefetch_bloom(buf[i].uhash);
            }
            for (int i = 0; i < n; ++i) {
                u8 d = buf[i].u_dom;
                if (__builtin_expect(d == DOM_SHORT, 1)) {
                    bool h = links.bloom_may_contain(buf[i].uhash);
                    hit[i] = h ? 1 : 0;
                    if (h) links.prefetch_short_alias(buf[i].uhash);
                } else if (d == DOM_HEX32) {
                    bool h = links.bloom_may_contain(buf[i].uhash);
                    hit[i] = h ? 1 : 0;
                    if (h) links.prefetch_hex_alias(buf[i].uhash);
                } else {
                    hit[i] = 1;
                }
            }
        }

        const u32 pmask = P - 1u;
        const u32 dsmask = DS - 1u;
        const u32 dlmask = DL - 1u;

        for (int i = 0; i < n; ++i) {
            const PendEvent& pe = buf[i];
            UserKey canon{pe.u_lo, pe.u_hi, pe.u_dom};
            if (check_links && hit[i]) {
                if (__builtin_expect(canon.dom == DOM_SHORT, 1)) {
                    canon = links.resolve_short_hit(canon.lo, pe.uhash);
                } else if (canon.dom == DOM_HEX32) {
                    canon = links.resolve_hex_hit(canon.hi, canon.lo, pe.uhash);
                } else {
                    auto it = links.exotic_aliases.find(u32(canon.lo));
                    if (it != links.exotic_aliases.end()) canon = it->second;
                }
            }

            u64 ch = (canon.dom == pe.u_dom && canon.lo == pe.u_lo && canon.hi == pe.u_hi)
                     ? pe.uhash : hash_user_key(canon);
            u32 p = u32(ch >> 32) & pmask;

            u32 loc;
            if (__builtin_expect(canon.dom == DOM_SHORT, 1)) {
                loc = w.emit_short_p(p, canon.lo, pe.ts_rel);
            } else if (canon.dom == DOM_HEX32) {
                loc = w.emit_long_p(p, canon.hi, canon.lo, pe.ts_rel);
            } else {
                loc = w.emit_exotic_p(p, u32(canon.lo), pe.ts_rel);
            }

            if (__builtin_expect(pe.price != 0, 0)) {
                w.emit_purch(p, {canon.lo, canon.hi, pe.ts_rel, pe.price, loc, pe.cat, canon.dom, 0});
            }

            if (__builtin_expect(pe.eid_dom == DOM_SHORT, 1)) {
                u32 q = u32(wyhash64(pe.eid_lo) >> 32) & dsmask;
                w.emit_short_e(q, {pe.eid_lo, pe.ing_sec, loc});
            } else {
                u32 q = u32(wyhash128(pe.eid_hi, pe.eid_lo) >> 32) & dlmask;
                w.emit_long_e(q, {pe.eid_hi, pe.eid_lo, pe.ing_sec, loc});
            }
        }
    }

    void flush_clicks(Worker& w, PendClick* buf, int n) {
        if (n == 0) return;
        const bool check_links = links.has_aliases;
        u8 hit[64];

        if (check_links) {
            for (int i = 0; i < n; ++i) {
                if (buf[i].raw_user.dom != DOM_EXOTIC) links.prefetch_bloom(buf[i].uhash);
            }
            for (int i = 0; i < n; ++i) {
                u8 d = buf[i].raw_user.dom;
                if (d == DOM_SHORT) {
                    bool h = links.bloom_may_contain(buf[i].uhash);
                    hit[i] = h ? 1 : 0;
                    if (h) links.prefetch_short_alias(buf[i].uhash);
                } else if (d == DOM_HEX32) {
                    bool h = links.bloom_may_contain(buf[i].uhash);
                    hit[i] = h ? 1 : 0;
                    if (h) links.prefetch_hex_alias(buf[i].uhash);
                } else {
                    hit[i] = 1;
                }
            }
        }

        const u32 pmask = P - 1u;
        for (int i = 0; i < n; ++i) {
            const PendClick& pc = buf[i];
            UserKey canon = pc.raw_user;
            if (check_links && hit[i]) {
                if (canon.dom == DOM_SHORT) {
                    canon = links.resolve_short_hit(canon.lo, pc.uhash);
                } else if (canon.dom == DOM_HEX32) {
                    canon = links.resolve_hex_hit(canon.hi, canon.lo, pc.uhash);
                } else {
                    auto it = links.exotic_aliases.find(u32(canon.lo));
                    if (it != links.exotic_aliases.end()) canon = it->second;
                }
            }
            u64 ch = (canon.dom == pc.raw_user.dom && canon.lo == pc.raw_user.lo && canon.hi == pc.raw_user.hi)
                     ? pc.uhash : hash_user_key(canon);
            u32 p = u32(ch >> 32) & pmask;
            u32 camp_dom = (u32(canon.dom) << 30) | pc.camp;
            w.emit_click(p, {canon.lo, canon.hi, pc.ts_rel, camp_dom, pc.file_len, pc.cid_off});
        }
    }

    bool use_pread_events = false;

    static inline size_t pread_all(int fd, u8* dst, size_t len, size_t off) {
        size_t got_tot = 0;
        while (got_tot < len) {
            ssize_t r = ::pread(fd, dst + got_tot, len - got_tot, off_t(off + got_tot));
            if (r < 0) {
                if (errno == EINTR) continue;
                die("pread failed");
            }
            if (r == 0) break;
            got_tot += size_t(r);
        }
        return got_tot;
    }

    void parse_events_and_clicks(ThreadPool& pool) {
        struct Task { u8 kind; u32 file; size_t s, e; };
        std::vector<Task> tasks;
        tasks.reserve(4096);

        const size_t EV_CSZ = use_pread_events ? (512u << 10) : (2u << 20);
        for (u32 fi = 0; fi < event_files.size(); ++fi) {
            const MFile& mf = event_files[fi];
            if (mf.size == 0) continue;
            size_t body = 0;
            if (use_pread_events) {
                u8 hbuf[512];
                size_t hn = pread_all(mf.fd, hbuf, std::min<size_t>(mf.size, sizeof(hbuf)), 0);
                const u8* nl = static_cast<const u8*>(std::memchr(hbuf, '\n', hn));
                if (!nl) {
                    // Header longer than 512B (rare): scan forward
                    size_t off = hn;
                    while (off < mf.size && !nl) {
                        hn = pread_all(mf.fd, hbuf, std::min<size_t>(mf.size - off, sizeof(hbuf)), off);
                        nl = static_cast<const u8*>(std::memchr(hbuf, '\n', hn));
                        if (nl) { body = off + size_t(nl - hbuf) + 1; break; }
                        off += hn;
                    }
                    if (!nl) continue;
                } else {
                    body = size_t(nl - hbuf) + 1;
                }
            } else {
                const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
                if (!nl) continue;
                body = size_t(nl - mf.data) + 1;
            }
            if (body >= mf.size) continue;
            for (size_t off = body; off < mf.size; off += EV_CSZ) {
                tasks.push_back({0, fi, off, std::min(off + EV_CSZ, mf.size)});
            }
        }
        {
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
        }

        std::atomic<size_t> next_task{0};
        pool.run([&](int tid) {
            Worker& w = workers[tid];
            PendEvent ev_buf[64];
            PendClick ck_buf[64];
            int nev = 0, nck = 0;

            // Reusable 64-byte aligned pread buffer (512 KiB + 8 KiB tail overlap + 128 B zero guard)
            constexpr size_t OVERLAP = 8192;
            alignas(64) u8* rbuf = nullptr;
            if (use_pread_events) {
                rbuf = static_cast<u8*>(::aligned_alloc(64, EV_CSZ + OVERLAP + 128));
            }

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
                    // Rare exotic event_id: emit immediately so PendEvent stays compact!
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
                    const MFile& mf = event_files[tk.file];
                    if (use_pread_events) {
                        // Read [tk.s - 1 (if s>0), min(mf.size, tk.e + OVERLAP)) into 64B-aligned rbuf!
                        size_t r_off = (tk.s > 0) ? (tk.s - 1) : 0;
                        size_t r_end = std::min(mf.size, tk.e + OVERLAP);
                        size_t r_len = pread_all(mf.fd, rbuf, r_end - r_off, r_off);
                        std::memset(rbuf + r_len, 0, 128); // guard bytes for 64B SIMD loads!

                        size_t s = (tk.s > 0) ? 1u : 0u;
                        if (tk.s > 0 && rbuf[0] != '\n') {
                            const u8* nl = static_cast<const u8*>(std::memchr(rbuf + 1, '\n', r_len - 1));
                            if (!nl) continue;
                            s = size_t(nl - rbuf) + 1;
                        }
                        size_t target_e = tk.e - r_off;
                        if (s >= target_e) continue;
                        size_t e = target_e;
                        if (tk.e < mf.size && rbuf[e - 1] != '\n') {
                            const u8* nl = static_cast<const u8*>(std::memchr(rbuf + e, '\n', r_len - e));
                            if (nl) {
                                e = size_t(nl - rbuf) + 1;
                            } else {
                                // Line exceeded OVERLAP (extremely rare): fallback read full line
                                e = r_len;
                            }
                        }
                        scan_csv<7>(rbuf, s, e, handle_event_line);
                    } else {
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
                    }
                } else {
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
                }
            }
            flush_events(w, ev_buf, nev);
            flush_clicks(w, ck_buf, nck);
            w.flush_all_stages();
            if (rbuf) std::free(rbuf);
        });
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
                            for (u32 i = 0; i < bn; ++i) {
                                const ShortERec& e = b->r[i];
                                u32 idx = u32(wyhash64(e.ekey)) & mask;
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
    // Phase 3: Single-Hash L1 Grouping + Sort-Free 30-Min Binning +
    //          O(S+C) Sliding-Window Attribution & Assists + Parallel Parquet Write
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

    void sessions_and_write(ThreadPool& pool) {
        const int nt = pool.size();
        const u32 ncamp = cats.num_campaigns;
        const u32 ncats = u32(cats.categories.size());
        const bool use_binning = (num_bins > 0 && num_bins <= 1024);
        const u32 nbin_words = (num_bins + 63u) >> 6;
        const u32 wsm = ws_mod;

        std::atomic<u32> next_p{0};

        pool.run([&](int tid) {
            Worker& w = workers[tid];
            w.ud_cols.reserve(350000, 100000);

            // Compact 16-byte L1-resident Short local table!
            struct ShortLSlot {
                u64 key; // 0 = empty
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

            // Single-hash pass buffers: cache (pid, ts) so we NEVER hash events twice!
            std::vector<u32> tmp_pid, tmp_ts;
            std::vector<u32> ev_ts;
            std::vector<u32> tmp_pu_pid;
            std::vector<LocalPurch> tmp_pu, pu_arr;
            std::vector<u32> tmp_ck_pid;
            std::vector<LocalClick> tmp_ck, ck_arr;

            std::vector<Bin> bins(use_binning ? num_bins : 1, Bin{0xFFFFFFFFu, 0, 0, 0});
            std::vector<u64> bmap(use_binning ? nbin_words : 1, 0);

            std::vector<u32> s_start, s_nev;
            std::vector<u16> s_day;
            std::vector<u64> s_rev;
            std::vector<u8>  s_has_p;
            std::vector<u64> catday(size_t(num_days) * size_t(ncats), 0);

            std::vector<u32> ccnt(ncamp, 0);
            std::vector<u32> cact(ncamp, 0);

            CampDayStat* cd = w.cd_stats.data();
            CampDayStat* org_cd = cd + size_t(ncamp) * size_t(num_days);

            auto add_assist = [&](u32 c, u32 s_lo, u32 s_hi) {
                CampDayStat* row = cd + size_t(c) * size_t(num_days);
                for (u32 j = s_lo; j < s_hi;) {
                    u16 d = s_day[j];
                    u32 e = j + 1;
                    while (e < s_hi && s_day[e] == d) ++e;
                    row[d].assisted_sessions += (e - j);
                    j = e;
                }
            };

            auto click_id_sv = [&](const LocalClick& c) -> std::string_view {
                u32 fi = c.file_len >> 20;
                u32 len = c.file_len & 0xFFFFFu;
                return std::string_view(
                    reinterpret_cast<const char*>(click_files[fi].data + c.cid_off), len);
            };

            for (;;) {
                u32 p = next_p.fetch_add(1, std::memory_order_relaxed);
                if (p >= P) break;

                u32 est_sp = 0, est_lp = 0, est_ep = 0, est_pu = 0, est_ck = 0;
                for (int t = 0; t < nt; ++t) {
                    est_sp += workers[t].sp_parts[p].total;
                    est_lp += workers[t].lp_parts[p].total;
                    est_ep += workers[t].ep_parts[p].total;
                    est_pu += workers[t].pu_parts[p].total;
                    est_ck += workers[t].ck_parts[p].total;
                }
                u32 est_ev = est_sp + est_lp + est_ep;
                if (est_ev == 0) continue;

                // Size compact Short local hash table (16 B/slot -> 32-64 KB, fits in L1/L2!)
                u32 scap = 128;
                u32 swant = std::max<u32>(est_sp / 6u + 64u, 64u);
                while (scap < swant) scap <<= 1;
                if (stab.size() < scap) stab.resize(scap);
                std::memset(stab.data(), 0, size_t(scap) * sizeof(ShortLSlot));
                u32 smask = scap - 1u;
                u32 s_used = 0;

                // Tiny Other local hash table (for DOM_HEX32 + DOM_EXOTIC)
                u32 ocap = 64;
                u32 owant = std::max<u32>((est_lp + est_ep) / 6u + 32u, 32u);
                while (ocap < owant) ocap <<= 1;
                if (otab.size() < ocap) otab.resize(ocap);
                for (u32 i = 0; i < ocap; ++i) otab[i].used = 0;
                u32 omask = ocap - 1u;
                u32 o_used = 0;

                lkeys.clear();
                ev_cnt.clear();
                tmp_pid.clear();
                tmp_ts.clear();
                if (tmp_pid.capacity() < est_ev) {
                    tmp_pid.reserve(est_ev);
                    tmp_ts.reserve(est_ev);
                }

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

                auto get_or_add_short = [&](u64 key) -> u32 {
                    if (__builtin_expect(s_used * 2 >= scap, 0)) grow_stab();
                    u32 idx = u32(hash_short(key)) & smask;
                    for (;;) {
                        ShortLSlot& s = stab[idx];
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

                auto find_any = [&](u64 lo, u64 hi, u8 dom) -> i32 {
                    if (__builtin_expect(dom == DOM_SHORT, 1)) {
                        u32 idx = u32(hash_short(lo)) & smask;
                        for (;;) {
                            const ShortLSlot& s = stab[idx];
                            if (s.key == 0) return -1;
                            if (s.key == lo) return i32(s.pid);
                            idx = (idx + 1u) & smask;
                        }
                    }
                    UserKey k{lo, hi, dom};
                    u32 idx = u32(hash_user_key(k)) & omask;
                    for (;;) {
                        const OtherLSlot& s = otab[idx];
                        if (!s.used) return -1;
                        if (s.key == k) return i32(s.pid);
                        idx = (idx + 1u) & omask;
                    }
                };

                // SINGLE HASH PASS over all surviving events!
                for (int t = 0; t < nt; ++t) {
                    for (ShortPBlock* b = workers[t].sp_parts[p].head; b; b = b->next) {
                        const u32 ng = b->ngroups;
                        for (u32 g = 0; g < ng; ++g) {
                            const ShortPGroup5& gr = b->g[g];
                            const u32 gc = gr.count;
                            for (u32 i = 0; i < gc; ++i) {
                                u32 ts = gr.ts[i];
                                if (__builtin_expect(ts == DEAD_TS, 0)) continue;
                                u32 pid = get_or_add_short(gr.ukey[i]);
                                ev_cnt[pid]++;
                                tmp_pid.push_back(pid);
                                tmp_ts.push_back(ts);
                            }
                        }
                    }
                    for (LongPBlock* b = workers[t].lp_parts[p].head; b; b = b->next) {
                        const u32 bn = b->n;
                        for (u32 i = 0; i < bn; ++i) {
                            u32 ts = b->r[i].ts;
                            if (__builtin_expect(ts == DEAD_TS, 0)) continue;
                            u32 pid = get_or_add_other({b->r[i].lo, b->r[i].hi, DOM_HEX32});
                            ev_cnt[pid]++;
                            tmp_pid.push_back(pid);
                            tmp_ts.push_back(ts);
                        }
                    }
                    for (ExoticPBlock* b = workers[t].ep_parts[p].head; b; b = b->next) {
                        const u32 bn = b->n;
                        for (u32 i = 0; i < bn; ++i) {
                            u32 ts = b->r[i].ts;
                            if (__builtin_expect(ts == DEAD_TS, 0)) continue;
                            u32 pid = get_or_add_other({b->r[i].eid, 0, DOM_EXOTIC});
                            ev_cnt[pid]++;
                            tmp_pid.push_back(pid);
                            tmp_ts.push_back(ts);
                        }
                    }
                }
                const u32 np = u32(lkeys.size());
                if (np == 0) continue;

                // Zero-hashing counting scatter of events!
                const u32 total_live_ev = u32(tmp_pid.size());
                ev_off.resize(np + 1);
                {
                    u32 run = 0;
                    for (u32 i = 0; i < np; ++i) { ev_off[i] = run; run += ev_cnt[i]; }
                    ev_off[np] = run;
                }
                if (ev_ts.size() < total_live_ev) ev_ts.resize(total_live_ev);
                std::memcpy(ev_cnt.data(), ev_off.data(), np * sizeof(u32));
                const u32* tp_ptr = tmp_pid.data();
                const u32* tt_ptr = tmp_ts.data();
                u32* ec_ptr = ev_cnt.data();
                u32* et_ptr = ev_ts.data();
                for (u32 i = 0; i < total_live_ev; ++i) {
                    et_ptr[ec_ptr[tp_ptr[i]]++] = tt_ptr[i];
                }

                // Single-hash pass for live purchases
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

                // Single-hash pass for live clicks
                ck_cnt.assign(np, 0);
                tmp_ck_pid.clear();
                tmp_ck.clear();
                if (est_ck > 0) {
                    for (int t = 0; t < nt; ++t) {
                        for (ClickBlock* b = workers[t].ck_parts[p].head; b; b = b->next) {
                            const u32 bn = b->n;
                            for (u32 i = 0; i < bn; ++i) {
                                const ClickRec& cr = b->r[i];
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

                // Process each person
                for (u32 pid = 0; pid < np; ++pid) {
                    u32 e0 = ev_off[pid], e1 = ev_off[pid + 1];
                    u32 ne = e1 - e0;
                    u32 u0 = pu_off[pid], u1 = pu_off[pid + 1];
                    u32 nu = u1 - u0;
                    u32 c0 = ck_off[pid], c1 = ck_off[pid + 1];
                    u32 nc = c1 - c0;

                    u32 ns = 0;
                    s_start.clear();
                    s_nev.clear();
                    s_day.clear();
                    s_rev.clear();
                    s_has_p.clear();

                    if (__builtin_expect(use_binning, 1)) {
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
                        for (u32 wi = 0; wi < nbin_words; ++wi) {
                            u64 m = BM[wi];
                            while (m) {
                                u32 b = (wi << 6) + u32(__builtin_ctzll(m));
                                m &= m - 1ULL;
                                Bin& e = B[b];
                                if (ns == 0 || (e.mn - prev_mx) > u32(SESSION_GAP_MS)) {
                                    s_start.push_back(e.mn);
                                    s_day.push_back(u16((e.mn + wsm) / u32(DAY_MS)));
                                    s_nev.push_back(0);
                                    s_rev.push_back(0);
                                    s_has_p.push_back(0);
                                    ns++;
                                }
                                s_nev[ns - 1] += e.cnt;
                                prev_mx = e.mx;
                                e.sid = ns - 1;
                            }
                        }

                        if (nu > 0) {
                            const LocalPurch* pu = pu_arr.data() + u0;
                            for (u32 i = 0; i < nu; ++i) {
                                u32 sid = B[pu[i].ts / u32(SESSION_GAP_MS)].sid;
                                s_rev[sid] += pu[i].price;
                                s_has_p[sid] = 1;
                                catday[size_t(s_day[sid]) * size_t(ncats) + pu[i].cat] += pu[i].price;
                            }
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
                    } else {
                        u32* ts = ev_ts.data() + e0;
                        std::sort(ts, ts + ne);
                        u32 prev = 0;
                        for (u32 i = 0; i < ne; ++i) {
                            u32 v = ts[i];
                            if (ns == 0 || (v - prev) > u32(SESSION_GAP_MS)) {
                                s_start.push_back(v);
                                s_day.push_back(u16(floor_div_day(window_start + i64(v)) - base_day));
                                s_nev.push_back(0);
                                s_rev.push_back(0);
                                s_has_p.push_back(0);
                                ns++;
                            }
                            s_nev[ns - 1]++;
                            prev = v;
                        }
                        if (nu > 0) {
                            const LocalPurch* pu = pu_arr.data() + u0;
                            for (u32 i = 0; i < nu; ++i) {
                                auto it = std::upper_bound(s_start.begin(), s_start.end(), pu[i].ts);
                                u32 sid = u32((it - s_start.begin()) - 1);
                                s_rev[sid] += pu[i].price;
                                s_has_p[sid] = 1;
                                catday[size_t(s_day[sid]) * size_t(ncats) + pu[i].cat] += pu[i].price;
                            }
                        }
                    }

                    u32 active_days = 0;
                    for (u32 i = 0; i < ns;) {
                        u16 d = s_day[i];
                        u32 d_sess = 0, d_ev = 0, d_ps = 0;
                        i64 d_rev = 0;
                        do {
                            d_sess++;
                            d_ev += s_nev[i];
                            d_rev += i64(s_rev[i]);
                            d_ps += s_has_p[i];
                            i++;
                        } while (i < ns && s_day[i] == d);

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
                            CampDayStat& x = org_cd[d];
                            x.sessions += d_sess;
                            x.purchasing_sessions += d_ps;
                            x.revenue_cents += d_rev;
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
                        u32 st = s_start[s] + u32(WEEK_MS);
                        while (hi < nc && ck[hi].ts <= st) {
                            u32 c = ck[hi].camp;
                            if (ccnt[c]++ == 0) cact[c] = s;
                            hi++;
                        }
                        while (lo < hi && ck[lo].ts + u32(WEEK_MS) < st) {
                            u32 c = ck[lo].camp;
                            if (--ccnt[c] == 0) add_assist(c, cact[c], s);
                            lo++;
                        }
                        u32 cr = (hi > lo) ? ck[hi - 1].camp : ncamp;
                        CampDayStat& x = cd[size_t(cr) * size_t(num_days) + s_day[s]];
                        x.sessions++;
                        x.purchasing_sessions += s_has_p[s];
                        x.revenue_cents += i64(s_rev[s]);
                    }
                    while (lo < hi) {
                        u32 c = ck[lo].camp;
                        if (--ccnt[c] == 0) add_assist(c, cact[c], ns);
                        lo++;
                    }
                }
            }

            if (!w.ud_cols.day.empty()) {
                std::string path = out_dir + "/user_daily/part-" + std::to_string(tid) + ".parquet";
                w.ud_cols.write_parquet(path, cats.categories);
            }
        });

        bool any_ud = false;
        for (int t = 0; t < nt; ++t) {
            if (!workers[t].ud_cols.day.empty()) { any_ud = true; break; }
        }
        if (!any_ud) {
            workers[0].ud_cols.write_parquet(out_dir + "/user_daily/part-0.parquet", cats.categories);
        }
    }

    void write_campaign_daily() {
        const u32 ncamp_plus1 = cats.num_campaigns + 1u;
        const size_t total = size_t(ncamp_plus1) * size_t(num_days);
        std::vector<CampDayStat> merged(total);
        for (int t = 0; t < int(workers.size()); ++t) {
            const auto& src = workers[t].cd_stats;
            for (size_t i = 0; i < total; ++i) {
                merged[i].sessions            += src[i].sessions;
                merged[i].purchasing_sessions += src[i].purchasing_sessions;
                merged[i].assisted_sessions   += src[i].assisted_sessions;
                merged[i].revenue_cents       += src[i].revenue_cents;
            }
        }

        std::vector<pq::CampaignDailyRow> rows;
        rows.reserve(total / 2 + 16);
        for (u32 c = 0; c < ncamp_plus1; ++c) {
            for (u32 d = 0; d < num_days; ++d) {
                const CampDayStat& st = merged[size_t(c) * size_t(num_days) + d];
                if (st.sessions > 0 || st.assisted_sessions > 0) {
                    rows.push_back({
                        c,
                        i32(base_day + d),
                        i32(st.sessions),
                        i32(st.purchasing_sessions),
                        st.revenue_cents,
                        i32(st.assisted_sessions),
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

        const char* env_io = std::getenv("HYPERION_IO");
        use_pread_events = (env_io && std::strcmp(env_io, "pread") == 0);

        size_t total_ev_bytes = 0;
        link_files.resize(lpaths.size());
        for (size_t i = 0; i < lpaths.size(); ++i) {
            if (!link_files[i].open_map(lpaths[i])) die("open link file");
            link_files[i].close_fd();
        }
        event_files.resize(epaths.size());
        for (size_t i = 0; i < epaths.size(); ++i) {
            if (use_pread_events) {
                if (!event_files[i].open_fd(epaths[i])) die("open event file");
            } else {
                if (!event_files[i].open_map(epaths[i])) die("open event file");
                event_files[i].close_fd();
            }
            total_ev_bytes += event_files[i].size;
        }
        click_files.resize(cpaths.size());
        for (size_t i = 0; i < cpaths.size(); ++i) {
            if (!click_files[i].open_map(cpaths[i])) die("open click file");
            click_files[i].close_fd();
        }

        // L1/L2 cache-tuned partition counts (overridable via env for A/B sweeps)
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

        // Overlapped catalogs + parallel lock-free links DSU + Bloom + 16B Alias Table!
        links.build_with_catalogs(link_files, cats, data_dir, pool, user_exotic);
        prof.mark("cat+links+bloom");

        workers.resize(eff_ncpu);
        for (int t = 0; t < eff_ncpu; ++t) {
            workers[t].init(t, P, DS, DL, cats.num_campaigns + 1u, num_days);
        }

        parse_events_and_clicks(pool);
        prof.mark("parse+scatter");

        dedup_phase(pool);
        prof.mark("dedup");

        sessions_and_write(pool);
        prof.mark("sessions+user_pq");

        write_campaign_daily();
        prof.mark("campaign_pq");

        if (eff_ncpu > 1 && total_ev_bytes > (256u << 20)) {
            u32 nreg = std::min<u32>(g_regions.count.load(std::memory_order_relaxed), 4096u);
            std::atomic<u32> ri{0};
            pool.run([&](int) {
                for (;;) {
                    u32 i = ri.fetch_add(1, std::memory_order_relaxed);
                    if (i >= nreg) break;
                    ::madvise(g_regions.regs[i].ptr, g_regions.regs[i].len, MADV_DONTNEED);
                }
            });
            prof.mark("madv_release");
        }

        pool.detach_all();
        _exit(0);
    }
};

} // namespace hyp

int main(int argc, char** argv) {
    static hyp::Pipeline pipe;
    pipe.run(argc, argv);
    return 0;
}
