#pragma once

#include "common.hpp"
#include "csv_time.hpp"
#include "keys.hpp"
#include <algorithm>
#include <dirent.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace HYP_NS {

inline std::vector<std::string> list_csv_files(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return out;
    while (struct dirent* e = ::readdir(d)) {
        const char* name = e->d_name;
        size_t len = std::strlen(name);
        if (len > 4 && std::memcmp(name + len - 4, ".csv", 4) == 0) {
            out.push_back(dir + "/" + name);
        }
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

inline void load_manifest(const std::string& path, i64& window_start, i64& window_end) {
    MFile mf;
    if (!mf.open_map(path)) die("cannot open manifest.json");
    std::string_view s(reinterpret_cast<const char*>(mf.data), mf.size);
    auto extract = [&](const char* key) -> i64 {
        size_t k = s.find(key);
        if (k == std::string_view::npos) die("manifest missing key");
        size_t c = s.find(':', k + std::strlen(key));
        if (c == std::string_view::npos) die("manifest syntax");
        size_t q1 = s.find('"', c + 1);
        if (q1 == std::string_view::npos) die("manifest quote1");
        size_t q2 = s.find('"', q1 + 1);
        if (q2 == std::string_view::npos) die("manifest quote2");
        i64 ts = parse_ts_general(reinterpret_cast<const u8*>(s.data() + q1 + 1), q2 - q1 - 1);
        if (ts == INVALID_TS) die("manifest bad timestamp");
        return ts;
    };
    window_start = extract("\"window_start\"");
    window_end = extract("\"window_end\"");
    mf.close_fd();
}

// ============================================================================
// Zero-Allocation Direct-Indexed Catalogs (products.csv & campaigns.csv)
// ============================================================================

struct Catalogs {
    std::vector<std::string> categories;
    u16 unknown_cat = 0;

    static constexpr u32 PROD_FAST_MAX = 65536;
    u16 prod_fast[PROD_FAST_MAX];
    std::unordered_map<std::string, u16> prod_slow;
    bool has_slow_prods = false;

    std::vector<std::string> camp_ids;
    std::vector<std::string> camp_channels;
    u32 num_campaigns = 0;

    static constexpr u32 CAMP_FAST_MAX = 4096;
    i32 camp_fast[CAMP_FAST_MAX];
    std::unordered_map<std::string, i32> camp_slow;
    bool has_slow_camps = false;

    static inline bool parse_prefixed_uint(const u8* p, size_t n, u32 prefix_len, u32 max_v, u32& out) {
        if (n <= prefix_len || n > prefix_len + 6) return false;
        const u8* d = p + prefix_len;
        size_t dn = n - prefix_len;
        if (d[0] < '1' || d[0] > '9') return false;
        u32 v = u32(d[0] - '0');
        for (size_t i = 1; i < dn; ++i) {
            u32 c = u32(d[i] - '0');
            if (c >= 10u) return false;
            v = v * 10u + c;
        }
        if (v >= max_v) return false;
        out = v;
        return true;
    }

    void load_products(const std::string& path) {
        MFile mf;
        if (!mf.open_map(path)) die("cannot open products.csv");

        std::vector<std::string> raw_cats;
        raw_cats.reserve(32);
        raw_cats.push_back("unknown");

        auto find_or_add_cat = [&](const u8* p, u32 n) -> u16 {
            for (u16 i = 0; i < raw_cats.size(); ++i) {
                if (raw_cats[i].size() == n && std::memcmp(raw_cats[i].data(), p, n) == 0) return i;
            }
            u16 id = u16(raw_cats.size());
            raw_cats.emplace_back(reinterpret_cast<const char*>(p), n);
            return id;
        };

        std::memset(prod_fast, 0, sizeof(prod_fast));
        std::vector<std::pair<std::string, u16>> slow_tmp;
        has_slow_prods = false;

        const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
        if (nl) {
            size_t body = size_t(nl - mf.data) + 1;
            scan_csv<2>(mf.data, body, mf.size, [&](const Field fl[2]) {
                if (__builtin_expect(fl[0].n == 0, 0)) return;
                u16 raw_cid = find_or_add_cat(fl[1].p, fl[1].n);
                u32 num = 0;
                if (__builtin_expect(fl[0].n >= 2 && fl[0].p[0] == 'p' &&
                    parse_prefixed_uint(fl[0].p, fl[0].n, 1, PROD_FAST_MAX, num), 1)) {
                    prod_fast[num] = raw_cid;
                } else {
                    slow_tmp.emplace_back(std::string(reinterpret_cast<const char*>(fl[0].p), fl[0].n), raw_cid);
                    has_slow_prods = true;
                }
            });
        }
        mf.close_fd();

        const u16 nc = u16(raw_cats.size());
        std::vector<u16> order(nc);
        for (u16 i = 0; i < nc; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](u16 a, u16 b) { return raw_cats[a] < raw_cats[b]; });

        std::vector<u16> raw_to_sorted(nc);
        categories.resize(nc);
        for (u16 s = 0; s < nc; ++s) {
            categories[s] = std::move(raw_cats[order[s]]);
            raw_to_sorted[order[s]] = s;
        }
        unknown_cat = raw_to_sorted[0];

        for (u32 i = 0; i < PROD_FAST_MAX; ++i) prod_fast[i] = raw_to_sorted[prod_fast[i]];
        prod_slow.clear();
        for (auto& kv : slow_tmp) prod_slow.emplace(std::move(kv.first), raw_to_sorted[kv.second]);
    }

    void load_campaigns(const std::string& path) {
        MFile mf;
        if (!mf.open_map(path)) die("cannot open campaigns.csv");
        for (u32 i = 0; i < CAMP_FAST_MAX; ++i) camp_fast[i] = -1;
        has_slow_camps = false;
        camp_ids.clear();
        camp_channels.clear();
        camp_ids.reserve(2500);
        camp_channels.reserve(2500);

        const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
        if (nl) {
            size_t body = size_t(nl - mf.data) + 1;
            scan_csv<2>(mf.data, body, mf.size, [&](const Field fl[2]) {
                if (__builtin_expect(fl[0].n == 0, 0)) return;
                i32 idx = i32(camp_ids.size());
                u32 num = 0;
                if (__builtin_expect(fl[0].n >= 3 && fl[0].p[0] == 'c' && fl[0].p[1] == '_' &&
                    parse_prefixed_uint(fl[0].p, fl[0].n, 2, CAMP_FAST_MAX, num), 1)) {
                    camp_fast[num] = idx;
                } else {
                    camp_slow.emplace(std::string(reinterpret_cast<const char*>(fl[0].p), fl[0].n), idx);
                    has_slow_camps = true;
                }
                camp_ids.emplace_back(reinterpret_cast<const char*>(fl[0].p), fl[0].n);
                camp_channels.emplace_back(reinterpret_cast<const char*>(fl[1].p), fl[1].n);
            });
        }
        mf.close_fd();
        num_campaigns = u32(camp_ids.size());
        camp_ids.push_back("organic");
        camp_channels.push_back("organic");
    }

    inline u16 lookup_product(const u8* p, u32 n) const {
        if (__builtin_expect(n >= 2 && p[0] == 'p', 1)) {
            u32 num = 0;
            if (__builtin_expect(parse_prefixed_uint(p, n, 1, PROD_FAST_MAX, num), 1)) {
                return prod_fast[num];
            }
        }
        if (__builtin_expect(!has_slow_prods, 1)) return unknown_cat;
        auto it = prod_slow.find(std::string(reinterpret_cast<const char*>(p), n));
        return it != prod_slow.end() ? it->second : unknown_cat;
    }

    inline i32 lookup_campaign(const u8* p, u32 n) const {
        if (__builtin_expect(n >= 3 && p[0] == 'c' && p[1] == '_', 1)) {
            u32 num = 0;
            if (__builtin_expect(parse_prefixed_uint(p, n, 2, CAMP_FAST_MAX, num), 1)) {
                return camp_fast[num];
            }
        }
        if (__builtin_expect(!has_slow_camps, 1)) return -1;
        auto it = camp_slow.find(std::string(reinterpret_cast<const char*>(p), n));
        return it != camp_slow.end() ? it->second : -1;
    }
};

// ============================================================================
// Fused 1-Pass Parallel Lock-Free DSU + 1 MB Blocked Bloom + Compact Alias Table
// ============================================================================

struct ShortAliasSlot {
    u64 key;        // 0 = empty
    u64 canon_enc;  // bit 63 == 0: DOM_SHORT key; bit 63 == 1: index into canon_overflow
};

struct HexAliasSlot {
    u64 hi;
    u64 lo;
    UserKey canon;
    std::atomic<u8> used;
};

struct LinkResolver {
    bool has_aliases = false;

    u64* bloom = nullptr;
    u32 bloom_mask = 0;

    ShortAliasSlot* short_tab = nullptr;
    u32 short_mask = 0;
    UserKey* canon_overflow = nullptr;

    HexAliasSlot* hex_tab = nullptr;
    u32 hex_mask = 0;

    std::unordered_map<u32, UserKey> exotic_aliases;

    static inline u64 bloom_bits(u64 h) {
        return (1ULL << (h & 63u)) |
               (1ULL << ((h >> 6) & 63u)) |
               (1ULL << ((h >> 12) & 63u));
    }

    inline bool bloom_may_contain(u64 h) const {
        u64 w = bloom[u32(h >> 24) & bloom_mask];
        u64 m = bloom_bits(h);
        return (w & m) == m;
    }

    inline void prefetch_bloom(u64 h) const {
        __builtin_prefetch(&bloom[u32(h >> 24) & bloom_mask], 0, 3);
    }

    inline void prefetch_short_alias(u64 h) const {
        __builtin_prefetch(&short_tab[u32(h) & short_mask], 0, 1);
    }

    inline void prefetch_hex_alias(u64 h) const {
        __builtin_prefetch(&hex_tab[u32(h) & hex_mask], 0, 1);
    }

    inline UserKey resolve_short_hit(u64 key, u64 h) const {
        if (__builtin_expect(!short_tab, 0)) return {key, 0, DOM_SHORT};
        u32 idx = u32(h) & short_mask;
        for (;;) {
            const ShortAliasSlot& s = short_tab[idx];
            if (s.key == 0) return {key, 0, DOM_SHORT};
            if (s.key == key) {
                if (__builtin_expect((s.canon_enc >> 63) == 0, 1)) {
                    return {s.canon_enc, 0, DOM_SHORT};
                }
                return canon_overflow[s.canon_enc & 0x7FFFFFFFFFFFFFFFULL];
            }
            idx = (idx + 1u) & short_mask;
        }
    }

    inline UserKey resolve_hex_hit(u64 hi, u64 lo, u64 h) const {
        if (__builtin_expect(!hex_tab, 0)) return {lo, hi, DOM_HEX32};
        u32 idx = u32(h) & hex_mask;
        for (;;) {
            const HexAliasSlot& s = hex_tab[idx];
            if (!s.used.load(std::memory_order_relaxed)) return {lo, hi, DOM_HEX32};
            if (s.hi == hi && s.lo == lo) return s.canon;
            idx = (idx + 1u) & hex_mask;
        }
    }

    void build_with_catalogs(const std::vector<MFile>& link_files,
                             Catalogs& cats, const std::string& data_dir,
                             ThreadPool& pool, ExoticInterner& ex,
                             const Profiler& prof) {
        const int nt = pool.size();
        struct LChunk { u32 file; size_t s, e; };
        std::vector<LChunk> chunks;
        size_t total_link_bytes = 0;
        size_t sample_lines = 0, sample_bytes = 0;

        for (u32 fi = 0; fi < link_files.size(); ++fi) {
            const MFile& mf = link_files[fi];
            if (mf.size == 0) continue;
            const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
            if (!nl) continue;
            size_t body = size_t(nl - mf.data) + 1;
            if (body >= mf.size) continue;
            total_link_bytes += (mf.size - body);

            // Sample up to 256 lines from first non-empty file to estimate avg line width accurately
            if (sample_lines == 0) {
                size_t cur = body;
                while (cur < mf.size && sample_lines < 256) {
                    const u8* eol = static_cast<const u8*>(std::memchr(mf.data + cur, '\n', mf.size - cur));
                    if (!eol) { sample_bytes += (mf.size - cur); sample_lines++; break; }
                    sample_bytes += size_t(eol - (mf.data + cur)) + 1;
                    sample_lines++;
                    cur = size_t(eol - mf.data) + 1;
                }
            }

            constexpr size_t CSZ = 512u << 10;
            for (size_t off = body; off < mf.size; off += CSZ) {
                chunks.push_back({fi, off, std::min(off + CSZ, mf.size)});
            }
        }

        if (chunks.empty()) {
            cats.load_products(data_dir + "/products.csv");
            cats.load_campaigns(data_dir + "/campaigns.csv");
            has_aliases = false;
            return;
        }

        double t_alloc0 = prof.enabled ? now_sec() : 0.0;
        size_t avg_line = sample_lines > 0 ? std::max<size_t>(sample_bytes / sample_lines, 4u) : 24u;
        // Conservative estimate: each edge introduces at most ~1.5 distinct nodes (at most 2.0 in disjoint pairs!)
        // Size tables at ~3.0x est_edges for short (~0.45 load), ~0.6x for hex (~0.35 load)
        const size_t est_edges = std::max<size_t>(total_link_bytes / avg_line, 256u);
        u32 scap = 1024;
        while (scap < est_edges * 2u && scap < (1u << 27)) scap <<= 1;
        const u32 smask = scap - 1u;

        u32 hcap = 256;
        while (hcap < est_edges / 3u + 256u && hcap < (1u << 25)) hcap <<= 1;
        const u32 hmask = hcap - 1u;

        const u32 max_nodes = u32(std::min<size_t>(est_edges * 2u + size_t(nt) * 2048u + 8192u, 1u << 26));
        struct ShortNodeSlot {
            std::atomic<u64> key;
            std::atomic<u32> id1;
            u32 pad;
        };
        struct HexNodeSlot {
            std::atomic<u64> state;
            u64 hi;
            u64 lo;
            u32 id;
            u32 pad;
        };
        struct alignas(16) DsuNode {
            u64 lo;
            std::atomic<u32> parent;
            u8 dom;
            u8 pad[3];
        };
        static_assert(sizeof(DsuNode) == 16);

        AnonRegion tmp_regs[4];
        auto alloc_tmp = [&](int idx, size_t bytes) -> void* {
            return huge_alloc(bytes, true, &tmp_regs[idx].ptr, &tmp_regs[idx].len);
        };

        auto* s_nodes   = static_cast<ShortNodeSlot*>(alloc_tmp(0, size_t(scap) * sizeof(ShortNodeSlot)));
        auto* h_nodes   = static_cast<HexNodeSlot*>(alloc_tmp(1, size_t(hcap) * sizeof(HexNodeSlot)));
        auto* nodes     = static_cast<DsuNode*>(alloc_tmp(2, size_t(max_nodes) * sizeof(DsuNode)));
        const u32 max_hex = u32(std::min<size_t>(hcap, 1u << 24));
        auto* hex_vals  = static_cast<Hex128*>(alloc_tmp(3, size_t(max_hex) * sizeof(Hex128)));

        std::atomic<u32> hex_ctr{0};
        std::mutex exotic_mu;
        std::unordered_map<u32, u32> exotic_node_map;
        std::atomic<u32> node_ctr{0};
        std::atomic<u32> total_short_nodes{0};
        std::atomic<u32> total_hex_nodes{0};
        std::atomic<u32> total_merges{0};
        std::atomic<size_t> next_ch{0};

        double t_p1 = prof.enabled ? now_sec() : 0.0;
        if (prof.enabled) prof.note("links:alloc", (t_p1 - t_alloc0) * 1000.0);

        auto get_key = [&](u32 id) -> UserKey {
            const DsuNode& nd = nodes[id];
            u8 d = nd.dom;
            if (__builtin_expect(d == DOM_SHORT, 1)) return {nd.lo, 0, DOM_SHORT};
            if (d == DOM_HEX32) {
                Hex128 h = hex_vals[nd.lo];
                return {h.lo, h.hi, DOM_HEX32};
            }
            return {nd.lo, 0, DOM_EXOTIC};
        };

        auto find_lf = [&](u32 x) -> u32 {
            while (true) {
                u32 p = nodes[x].parent.load(std::memory_order_relaxed);
                if (p == x) return x;
                u32 gp = nodes[p].parent.load(std::memory_order_relaxed);
                if (gp != p) {
                    nodes[x].parent.compare_exchange_weak(p, gp, std::memory_order_relaxed, std::memory_order_relaxed);
                }
                x = gp;
            }
        };

        // Fused Pass 1: Load catalogs on thread 0 while ALL threads parse links, intern nodes, AND union lock-free!
        pool.run([&](int tid) {
            if (tid == 0) {
                cats.load_products(data_dir + "/products.csv");
                cats.load_campaigns(data_dir + "/campaigns.csv");
            }
            u32 loc_cur = 0, loc_end = 0;
            u32 loc_short = 0, loc_hex = 0, loc_merges = 0;

            auto alloc_id = [&]() -> u32 {
                if (__builtin_expect(loc_cur == loc_end, 0)) {
                    loc_cur = node_ctr.fetch_add(512, std::memory_order_relaxed);
                    loc_end = loc_cur + 512;
                }
                return loc_cur++;
            };

            auto intern_node = [&](const UserKey& k, u64 kh) -> u32 {
                if (__builtin_expect(k.dom == DOM_SHORT, 1)) {
                    u32 idx = u32(kh) & smask;
                    for (;;) {
                        u64 cur = s_nodes[idx].key.load(std::memory_order_relaxed);
                        if (cur == k.lo) {
                            u32 id1;
                            while ((id1 = s_nodes[idx].id1.load(std::memory_order_acquire)) == 0)
                                __builtin_ia32_pause();
                            return id1 - 1u;
                        }
                        if (cur == 0) {
                            u64 exp = 0;
                            if (s_nodes[idx].key.compare_exchange_strong(exp, k.lo,
                                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                                u32 id = alloc_id();
                                ++loc_short;
                                nodes[id].lo = k.lo;
                                nodes[id].dom = DOM_SHORT;
                                nodes[id].parent.store(id, std::memory_order_relaxed);
                                s_nodes[idx].id1.store(id + 1u, std::memory_order_release);
                                return id;
                            }
                            if (exp == k.lo) {
                                u32 id1;
                                while ((id1 = s_nodes[idx].id1.load(std::memory_order_acquire)) == 0)
                                    __builtin_ia32_pause();
                                return id1 - 1u;
                            }
                        }
                        idx = (idx + 1u) & smask;
                    }
                }
                if (k.dom == DOM_HEX32) {
                    u32 idx = u32(kh) & hmask;
                    for (;;) {
                        u64 st = h_nodes[idx].state.load(std::memory_order_acquire);
                        if (st == 2) {
                            if (h_nodes[idx].hi == k.hi && h_nodes[idx].lo == k.lo) return h_nodes[idx].id;
                            idx = (idx + 1u) & hmask;
                            continue;
                        }
                        if (st == 0) {
                            u64 exp = 0;
                            if (h_nodes[idx].state.compare_exchange_strong(exp, 1,
                                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                                u32 id = alloc_id();
                                u32 hx_idx = hex_ctr.fetch_add(1, std::memory_order_relaxed);
                                ++loc_hex;
                                hex_vals[hx_idx] = {k.hi, k.lo};
                                nodes[id].lo = hx_idx;
                                nodes[id].dom = DOM_HEX32;
                                nodes[id].parent.store(id, std::memory_order_relaxed);
                                h_nodes[idx].hi = k.hi;
                                h_nodes[idx].lo = k.lo;
                                h_nodes[idx].id = id;
                                h_nodes[idx].state.store(2, std::memory_order_release);
                                return id;
                            }
                            continue;
                        }
                        __builtin_ia32_pause();
                    }
                }
                std::lock_guard<std::mutex> lk(exotic_mu);
                auto it = exotic_node_map.find(u32(k.lo));
                if (it != exotic_node_map.end()) return it->second;
                u32 id = alloc_id();
                nodes[id].lo = k.lo;
                nodes[id].dom = DOM_EXOTIC;
                nodes[id].parent.store(id, std::memory_order_relaxed);
                exotic_node_map.emplace(u32(k.lo), id);
                return id;
            };

            UserKey pa[32], pb[32];
            u64 ha[32], hb[32];
            int nb = 0;
            auto flush_edges = [&]() {
                for (int i = 0; i < nb; ++i) {
                    if (pa[i].dom == DOM_SHORT) {
                        ha[i] = hash_short(pa[i].lo);
                        __builtin_prefetch(&s_nodes[u32(ha[i]) & smask], 1, 1);
                    } else if (pa[i].dom == DOM_HEX32) {
                        ha[i] = hash_hex128(pa[i].hi, pa[i].lo);
                        __builtin_prefetch(&h_nodes[u32(ha[i]) & hmask], 1, 1);
                    }
                    if (pb[i].dom == DOM_SHORT) {
                        hb[i] = hash_short(pb[i].lo);
                        __builtin_prefetch(&s_nodes[u32(hb[i]) & smask], 1, 1);
                    } else if (pb[i].dom == DOM_HEX32) {
                        hb[i] = hash_hex128(pb[i].hi, pb[i].lo);
                        __builtin_prefetch(&h_nodes[u32(hb[i]) & hmask], 1, 1);
                    }
                }
                for (int i = 0; i < nb; ++i) {
                    u32 ia = intern_node(pa[i], ha[i]);
                    u32 ib = intern_node(pb[i], hb[i]);
                    u32 ra = find_lf(ia);
                    u32 rb = find_lf(ib);
                    while (ra != rb) {
                        // Union by byte-wise key order so root IS ALWAYS the canonical smallest UserKey!
                        if (user_key_less(get_key(rb), get_key(ra), ex)) std::swap(ra, rb);
                        u32 expected = rb;
                        if (nodes[rb].parent.compare_exchange_weak(expected, ra,
                                std::memory_order_acq_rel, std::memory_order_acquire)) {
                            ++loc_merges;
                            break;
                        }
                        ra = find_lf(ra);
                        rb = find_lf(rb);
                    }
                }
                nb = 0;
            };

            for (;;) {
                size_t ci = next_ch.fetch_add(1, std::memory_order_relaxed);
                if (ci >= chunks.size()) break;
                const auto& ch = chunks[ci];
                const MFile& mf = link_files[ch.file];
                size_t s = ch.s;
                if (s > 0 && mf.data[s - 1] != '\n') {
                    const u8* nl = static_cast<const u8*>(std::memchr(mf.data + s, '\n', mf.size - s));
                    if (!nl) continue;
                    s = size_t(nl - mf.data) + 1;
                }
                if (s >= ch.e) continue;
                size_t e = ch.e;
                if (e < mf.size && mf.data[e - 1] != '\n') {
                    const u8* nl = static_cast<const u8*>(std::memchr(mf.data + e, '\n', mf.size - e));
                    e = nl ? size_t(nl - mf.data) + 1 : mf.size;
                }
                scan_csv<2>(mf.data, s, e, [&](const Field fl[2]) {
                    if (__builtin_expect(fl[0].n == 0 || fl[1].n == 0, 0)) return;
                    UserKey ka = encode_user_key(fl[0].p, fl[0].n, ex);
                    UserKey kb = encode_user_key(fl[1].p, fl[1].n, ex);
                    if (__builtin_expect(ka == kb, 0)) return;
                    pa[nb] = ka;
                    pb[nb] = kb;
                    if (++nb == 32) flush_edges();
                });
            }
            if (nb > 0) flush_edges();

            for (u32 u = loc_cur; u < loc_end; ++u) nodes[u].dom = 255;
            if (loc_short)  total_short_nodes.fetch_add(loc_short, std::memory_order_relaxed);
            if (loc_hex)    total_hex_nodes.fetch_add(loc_hex, std::memory_order_relaxed);
            if (loc_merges) total_merges.fetch_add(loc_merges, std::memory_order_relaxed);
        });

        double t_p2 = prof.enabled ? now_sec() : 0.0;
        if (prof.enabled) prof.note("links:pass1", (t_p2 - t_p1) * 1000.0);

        const u32 total_slots = node_ctr.load(std::memory_order_relaxed);
        const u32 n_merges = total_merges.load(std::memory_order_relaxed);
        if (total_slots == 0 || n_merges == 0) {
            has_aliases = false;
            return;
        }

        // Fused Pass 2+3: Since root is already the byte-wise smallest key per component,
        // AND n_merges is the EXACT number of non-root aliases across all domains,
        // we size Bloom + Compact Alias Tables tightly from n_merges in a SINGLE pass!
        has_aliases = true;
        u32 n_sn = total_short_nodes.load();
        u32 n_hn = total_hex_nodes.load();
        u32 max_aliases = std::max<u32>(n_merges, 64u);
        u32 bwords = 2048;
        while (u64(bwords) * 64ULL < u64(max_aliases) * 12ULL) bwords <<= 1;
        bloom_mask = bwords - 1u;
        bloom = static_cast<u64*>(huge_alloc(size_t(bwords) * sizeof(u64)));

        u32 stab_cap = 0, htab_cap = 0;
        if (n_sn > 0) {
            u32 max_sn_aliases = std::min(n_sn, max_aliases);
            stab_cap = 64;
            while (stab_cap < max_sn_aliases + (max_sn_aliases >> 2) + 64u) stab_cap <<= 1;
            short_mask = stab_cap - 1u;
            short_tab = static_cast<ShortAliasSlot*>(huge_alloc(size_t(stab_cap) * sizeof(ShortAliasSlot)));
            canon_overflow = static_cast<UserKey*>(huge_alloc(size_t(max_sn_aliases + 64u) * sizeof(UserKey)));
        }
        if (n_hn > 0) {
            u32 max_hn_aliases = std::min(n_hn, max_aliases);
            htab_cap = 64;
            while (htab_cap < max_hn_aliases + (max_hn_aliases >> 2) + 64u) htab_cap <<= 1;
            hex_mask = htab_cap - 1u;
            hex_tab = static_cast<HexAliasSlot*>(huge_alloc(size_t(htab_cap) * sizeof(HexAliasSlot)));
        }
        if (!thp_safe()) {
            if (bloom)     back_huge_window(bloom, size_t(bwords) * sizeof(u64));
            if (short_tab) back_huge_window(short_tab, size_t(stab_cap) * sizeof(ShortAliasSlot));
            if (hex_tab)   back_huge_window(hex_tab, size_t(htab_cap) * sizeof(HexAliasSlot));
        }

        std::atomic<u32> ovf_ctr{0};
        auto* abloom = reinterpret_cast<std::atomic<u64>*>(bloom);
        auto* ashort = reinterpret_cast<std::atomic<u64>*>(short_tab);
        std::mutex ex_mu;

        pool.run([&](int tid) {
            u32 lo = u32((u64(total_slots) * u64(tid)) / u64(nt));
            u32 hi = u32((u64(total_slots) * u64(tid + 1)) / u64(nt));
            struct PendS { u64 key; u64 cenc; u64 h; };
            PendS ps[32];
            int nps = 0;
            auto flush_ps = [&]() {
                for (int k = 0; k < nps; ++k) {
                    __builtin_prefetch(&abloom[u32(ps[k].h >> 24) & bloom_mask], 1, 3);
                    __builtin_prefetch(&short_tab[u32(ps[k].h) & short_mask], 1, 1);
                }
                for (int k = 0; k < nps; ++k) {
                    u64 h = ps[k].h;
                    abloom[u32(h >> 24) & bloom_mask].fetch_or(bloom_bits(h), std::memory_order_relaxed);
                    u32 idx = u32(h) & short_mask;
                    for (;;) {
                        u64 exp = 0;
                        if (ashort[idx * 2].compare_exchange_strong(exp, ps[k].key,
                                std::memory_order_relaxed, std::memory_order_relaxed)) {
                            short_tab[idx].canon_enc = ps[k].cenc;
                            break;
                        }
                        idx = (idx + 1u) & short_mask;
                    }
                }
                nps = 0;
            };

            for (u32 i = lo; i < hi; ++i) {
                const DsuNode& nd = nodes[i];
                u8 d = nd.dom;
                if (d == 255) continue;
                u32 b = i;
                while (true) {
                    u32 p = nodes[b].parent.load(std::memory_order_relaxed);
                    if (p == b) break;
                    b = p;
                }
                if (b == i) continue;
                if (__builtin_expect(d == DOM_SHORT, 1)) {
                    u64 akey = nd.lo;
                    u64 h = hash_short(akey);
                    u64 cenc;
                    const DsuNode& nb_node = nodes[b];
                    if (__builtin_expect(nb_node.dom == DOM_SHORT, 1)) {
                        cenc = nb_node.lo;
                    } else {
                        u32 oid = ovf_ctr.fetch_add(1, std::memory_order_relaxed);
                        canon_overflow[oid] = get_key(b);
                        cenc = (1ULL << 63) | u64(oid);
                    }
                    ps[nps++] = {akey, cenc, h};
                    if (nps == 32) flush_ps();
                } else if (d == DOM_HEX32) {
                    UserKey alias = get_key(i);
                    UserKey canon = get_key(b);
                    u64 h = hash_hex128(alias.hi, alias.lo);
                    abloom[u32(h >> 24) & bloom_mask].fetch_or(bloom_bits(h), std::memory_order_relaxed);
                    u32 idx = u32(h) & hex_mask;
                    for (;;) {
                        u8 exp = 0;
                        if (hex_tab[idx].used.compare_exchange_strong(exp, 1,
                                std::memory_order_relaxed, std::memory_order_relaxed)) {
                            hex_tab[idx].hi = alias.hi;
                            hex_tab[idx].lo = alias.lo;
                            hex_tab[idx].canon = canon;
                            break;
                        }
                        idx = (idx + 1u) & hex_mask;
                    }
                } else {
                    UserKey alias = get_key(i);
                    UserKey canon = get_key(b);
                    std::lock_guard<std::mutex> lk(ex_mu);
                    exotic_aliases[u32(alias.lo)] = canon;
                }
            }
            if (nps > 0) flush_ps();
        });

        double t_rel = prof.enabled ? now_sec() : 0.0;
        if (prof.enabled) {
            prof.note("links:pass2_fused", (t_rel - t_p2) * 1000.0);
            std::fprintf(stderr, "  [sub] links:stats             est_e=%zu scap=%u hcap=%u sn=%u hn=%u stab=%u htab=%u\n",
                         est_edges, scap, hcap, n_sn, n_hn, stab_cap, htab_cap);
        }
    }
};

} // namespace HYP_NS
