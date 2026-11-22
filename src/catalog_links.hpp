#pragma once

#include "common.hpp"
#include "csv_time.hpp"
#include "keys.hpp"
#include <algorithm>
#include <dirent.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace hyp {

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
// Direct-Indexed Catalogs (products.csv & campaigns.csv)
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
        std::vector<std::pair<std::string, std::string>> rows;
        rows.reserve(55000);
        const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
        if (nl) {
            size_t pos = size_t(nl - mf.data) + 1;
            Field fl[2];
            while (pos < mf.size) {
                const u8* p = mf.data + pos;
                const u8* e = static_cast<const u8*>(std::memchr(p, '\n', mf.size - pos));
                const u8* lend = e ? e : (mf.data + mf.size);
                pos = e ? size_t(e - mf.data) + 1 : mf.size;
                if (split_exact<2>(p, lend, fl) && fl[0].n > 0) {
                    rows.emplace_back(
                        std::string(reinterpret_cast<const char*>(fl[0].p), fl[0].n),
                        std::string(reinterpret_cast<const char*>(fl[1].p), fl[1].n));
                }
            }
        }
        mf.close_fd();

        categories.clear();
        categories.push_back("unknown");
        for (const auto& r : rows) categories.push_back(r.second);
        std::sort(categories.begin(), categories.end());
        categories.erase(std::unique(categories.begin(), categories.end()), categories.end());

        std::unordered_map<std::string, u16> cat_to_id;
        for (u16 i = 0; i < categories.size(); ++i) {
            cat_to_id[categories[i]] = i;
            if (categories[i] == "unknown") unknown_cat = i;
        }

        for (u32 i = 0; i < PROD_FAST_MAX; ++i) prod_fast[i] = unknown_cat;
        has_slow_prods = false;
        for (const auto& r : rows) {
            u16 cid = cat_to_id[r.second];
            u32 num = 0;
            if (r.first.size() >= 2 && r.first[0] == 'p' &&
                parse_prefixed_uint(reinterpret_cast<const u8*>(r.first.data()), r.first.size(), 1, PROD_FAST_MAX, num)) {
                prod_fast[num] = cid;
            } else {
                prod_slow[r.first] = cid;
                has_slow_prods = true;
            }
        }
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
            size_t pos = size_t(nl - mf.data) + 1;
            Field fl[2];
            while (pos < mf.size) {
                const u8* p = mf.data + pos;
                const u8* e = static_cast<const u8*>(std::memchr(p, '\n', mf.size - pos));
                const u8* lend = e ? e : (mf.data + mf.size);
                pos = e ? size_t(e - mf.data) + 1 : mf.size;
                if (split_exact<2>(p, lend, fl) && fl[0].n > 0) {
                    i32 idx = i32(camp_ids.size());
                    std::string cid(reinterpret_cast<const char*>(fl[0].p), fl[0].n);
                    std::string ch(reinterpret_cast<const char*>(fl[1].p), fl[1].n);
                    u32 num = 0;
                    if (cid.size() >= 3 && cid[0] == 'c' && cid[1] == '_' &&
                        parse_prefixed_uint(fl[0].p, fl[0].n, 2, CAMP_FAST_MAX, num)) {
                        camp_fast[num] = idx;
                    } else {
                        camp_slow[cid] = idx;
                        has_slow_camps = true;
                    }
                    camp_ids.push_back(std::move(cid));
                    camp_channels.push_back(std::move(ch));
                }
            }
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
// Parallel Lock-Free DSU + 1-Word Blocked Bloom + Compact 16B Alias Table
// ============================================================================

struct LinkEdge {
    UserKey a;
    UserKey b;
};

// Compact 16-byte Short alias slot (fits 4 slots per cache line, 16 MB at 20M!)
// canon_enc: if bit 63 == 0 -> DOM_SHORT key verbatim;
//            if bit 63 == 1 -> index into canon_overflow[] (DOM_HEX32 or DOM_EXOTIC)!
struct ShortAliasSlot {
    u64 key;        // 0 = empty
    u64 canon_enc;
};

struct HexAliasSlot {
    u64 hi;
    u64 lo;
    UserKey canon;
    u8 used;
};

struct LinkResolver {
    bool has_aliases = false;

    u64* bloom = nullptr;
    u32 bloom_mask = 0;

    ShortAliasSlot* short_tab = nullptr;
    u32 short_mask = 0;
    std::vector<UserKey> canon_overflow;

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
            if (!s.used) return {lo, hi, DOM_HEX32};
            if (s.hi == hi && s.lo == lo) return s.canon;
            idx = (idx + 1u) & hex_mask;
        }
    }

    // Build links DSU + Bloom + Alias tables in parallel across all cores,
    // overlapping catalog loading on thread 0!
    void build_with_catalogs(const std::vector<MFile>& link_files,
                             Catalogs& cats, const std::string& data_dir,
                             ThreadPool& pool, ExoticInterner& ex) {
        const int nt = pool.size();
        struct LChunk { u32 file; size_t s, e; };
        std::vector<LChunk> chunks;
        for (u32 fi = 0; fi < link_files.size(); ++fi) {
            const MFile& mf = link_files[fi];
            if (mf.size == 0) continue;
            const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
            if (!nl) continue;
            size_t body = size_t(nl - mf.data) + 1;
            if (body >= mf.size) continue;
            constexpr size_t CSZ = 512u << 10; // 512 KB chunks for good 8-way balance
            for (size_t off = body; off < mf.size; off += CSZ) {
                chunks.push_back({fi, off, std::min(off + CSZ, mf.size)});
            }
        }

        // Step 1: Parse links/*.csv in parallel while thread 0 loads products + campaigns!
        std::vector<std::vector<LinkEdge>> t_edges(nt);
        std::atomic<size_t> next_ch{0};
        pool.run([&](int tid) {
            if (tid == 0) {
                cats.load_products(data_dir + "/products.csv");
                cats.load_campaigns(data_dir + "/campaigns.csv");
            }
            auto& out = t_edges[tid];
            out.reserve(32768);
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
                    if (ka != kb) out.push_back({ka, kb});
                });
            }
        });

        size_t total_edges = 0;
        for (int t = 0; t < nt; ++t) total_edges += t_edges[t].size();
        if (total_edges == 0) {
            has_aliases = false;
            return;
        }

        // Step 2: Parallel Lock-Free Node Interning!
        // Size short node table to ~2.5x edges (at 890K edges -> 2M slots * 16B = 32 MB!)
        u32 scap = 1024;
        while (scap < total_edges * 3 && scap < (1u << 27)) scap <<= 1;
        const u32 smask = scap - 1u;

        // Hex/Exotic node table: only ~12.5% of nodes -> ~4x smaller!
        u32 hcap = 256;
        while (hcap < total_edges / 2 + 256 && hcap < (1u << 25)) hcap <<= 1;
        const u32 hmask = hcap - 1u;

        struct ShortNodeSlot {
            std::atomic<u64> key; // 0 = empty
            std::atomic<u32> id1; // dense_id + 1 (0 = being written)
            u32 pad;
        };
        struct HexNodeSlot {
            std::atomic<u64> state; // 0 = empty, 1 = locked, 2 = ready
            u64 hi;
            u64 lo;
            u32 id;
            u32 pad;
        };

        const u32 max_nodes = u32(total_edges * 2 + size_t(nt) * 1024 + 256);
        auto* s_nodes = static_cast<ShortNodeSlot*>(huge_alloc(size_t(scap) * sizeof(ShortNodeSlot)));
        auto* h_nodes = static_cast<HexNodeSlot*>(huge_alloc(size_t(hcap) * sizeof(HexNodeSlot)));
        auto* node_keys = static_cast<UserKey*>(huge_alloc(size_t(max_nodes) * sizeof(UserKey)));
        auto* parent = static_cast<std::atomic<u32>*>(huge_alloc(size_t(max_nodes) * sizeof(std::atomic<u32>)));
        auto* best   = static_cast<std::atomic<u32>*>(huge_alloc(size_t(max_nodes) * sizeof(std::atomic<u32>)));

        std::mutex exotic_mu;
        std::unordered_map<u32, u32> exotic_node_map;
        std::atomic<u32> node_ctr{0};
        std::atomic<u32> total_short_nodes{0};
        std::atomic<u32> total_hex_nodes{0};

        std::vector<std::vector<std::pair<u32, u32>>> t_iedges(nt);

        pool.run([&](int tid) {
            const auto& edges = t_edges[tid];
            auto& iedges = t_iedges[tid];
            iedges.resize(edges.size());
            u32 loc_short = 0, loc_hex = 0;

            // Batch-allocate dense node IDs in chunks of 512 (zero atomic contention!)
            u32 loc_cur = 0, loc_end = 0;
            auto alloc_id = [&]() -> u32 {
                if (__builtin_expect(loc_cur == loc_end, 0)) {
                    loc_cur = node_ctr.fetch_add(512, std::memory_order_relaxed);
                    loc_end = loc_cur + 512;
                }
                return loc_cur++;
            };

            auto intern_node = [&](const UserKey& k) -> u32 {
                if (__builtin_expect(k.dom == DOM_SHORT, 1)) {
                    u32 idx = u32(hash_short(k.lo)) & smask;
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
                                node_keys[id] = k;
                                parent[id].store(id, std::memory_order_relaxed);
                                best[id].store(id, std::memory_order_relaxed);
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
                    u32 idx = u32(hash_hex128(k.hi, k.lo)) & hmask;
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
                                ++loc_hex;
                                node_keys[id] = k;
                                parent[id].store(id, std::memory_order_relaxed);
                                best[id].store(id, std::memory_order_relaxed);
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
                // DOM_EXOTIC (<0.001% of links + fixtures)
                std::lock_guard<std::mutex> lk(exotic_mu);
                auto it = exotic_node_map.find(u32(k.lo));
                if (it != exotic_node_map.end()) return it->second;
                u32 id = alloc_id();
                node_keys[id] = k;
                parent[id].store(id, std::memory_order_relaxed);
                best[id].store(id, std::memory_order_relaxed);
                exotic_node_map.emplace(u32(k.lo), id);
                return id;
            };

            const size_t en = edges.size();
            for (size_t i = 0; i < en; ++i) {
                if (i + 12 < en) {
                    const auto& ea = edges[i + 12].a;
                    const auto& eb = edges[i + 12].b;
                    if (ea.dom == DOM_SHORT) __builtin_prefetch(&s_nodes[u32(hash_short(ea.lo)) & smask], 1, 1);
                    if (eb.dom == DOM_SHORT) __builtin_prefetch(&s_nodes[u32(hash_short(eb.lo)) & smask], 1, 1);
                }
                u32 ia = intern_node(edges[i].a);
                u32 ib = intern_node(edges[i].b);
                iedges[i] = {ia, ib};
            }
            // Mark unused trailing batch slots so Step 4 skips them
            for (u32 u = loc_cur; u < loc_end; ++u) {
                node_keys[u].dom = 255;
            }
            if (loc_short) total_short_nodes.fetch_add(loc_short, std::memory_order_relaxed);
            if (loc_hex)   total_hex_nodes.fetch_add(loc_hex, std::memory_order_relaxed);
        });

        const u32 total_slots = node_ctr.load(std::memory_order_relaxed);

        // Step 3: Parallel Lock-Free Union-Find (attach larger id under smaller id -> acyclic!)
        auto find_lf = [&](u32 x) -> u32 {
            while (true) {
                u32 p = parent[x].load(std::memory_order_relaxed);
                if (p == x) return x;
                u32 gp = parent[p].load(std::memory_order_relaxed);
                if (gp != p) {
                    parent[x].compare_exchange_weak(p, gp, std::memory_order_relaxed, std::memory_order_relaxed);
                }
                x = gp;
            }
        };

        pool.run([&](int tid) {
            for (const auto& [a, b] : t_iedges[tid]) {
                u32 ra = find_lf(a);
                u32 rb = find_lf(b);
                while (ra != rb) {
                    if (ra > rb) std::swap(ra, rb);
                    u32 expected = rb;
                    if (parent[rb].compare_exchange_weak(expected, ra,
                            std::memory_order_acq_rel, std::memory_order_acquire)) {
                        break;
                    }
                    ra = find_lf(ra);
                    rb = find_lf(rb);
                }
            }
        });

        // Step 4: Parallel path compression (in-place!) + find byte-wise smallest name per component!
        pool.run([&](int tid) {
            u32 lo = u32((u64(total_slots) * u64(tid)) / u64(nt));
            u32 hi = u32((u64(total_slots) * u64(tid + 1)) / u64(nt));
            for (u32 i = lo; i < hi; ++i) {
                if (node_keys[i].dom == 255) continue;
                u32 r = i;
                while (true) {
                    u32 p = parent[r].load(std::memory_order_relaxed);
                    if (p == r) break;
                    r = p;
                }
                parent[i].store(r, std::memory_order_relaxed);
                u32 cur_b = best[r].load(std::memory_order_relaxed);
                while (cur_b != i && user_key_less(node_keys[i], node_keys[cur_b], ex)) {
                    if (best[r].compare_exchange_weak(cur_b, i,
                            std::memory_order_acq_rel, std::memory_order_relaxed)) {
                        break;
                    }
                }
            }
        });

        // Size Bloom + Alias tables safely from exact node counts (zero extra barrier!)
        has_aliases = true;
        u32 n_sn = total_short_nodes.load();
        u32 n_hn = total_hex_nodes.load();
        u32 est_aliases = std::max<u32>((n_sn + n_hn) / 2u, 64u);
        u32 bwords = 4096;
        while (u64(bwords) * 64ULL < u64(est_aliases) * 20ULL) bwords <<= 1;
        bloom_mask = bwords - 1u;
        bloom = static_cast<u64*>(huge_alloc(size_t(bwords) * sizeof(u64)));

        if (n_sn > 0) {
            u32 stab_cap = 64;
            while (stab_cap < n_sn * 2u) stab_cap <<= 1;
            short_mask = stab_cap - 1u;
            short_tab = static_cast<ShortAliasSlot*>(huge_alloc(size_t(stab_cap) * sizeof(ShortAliasSlot)));
        }
        if (n_hn > 0) {
            u32 htab_cap = 64;
            while (htab_cap < n_hn * 2u) htab_cap <<= 1;
            hex_mask = htab_cap - 1u;
            hex_tab = static_cast<HexAliasSlot*>(huge_alloc(size_t(htab_cap) * sizeof(HexAliasSlot)));
        }

        // Step 5: Populate Bloom + Compact Alias Tables in parallel!
        std::mutex ovf_mu;
        auto* abloom = reinterpret_cast<std::atomic<u64>*>(bloom);
        auto* ashort = reinterpret_cast<std::atomic<u64>*>(short_tab);
        std::mutex hex_ex_mu;

        pool.run([&](int tid) {
            u32 lo = u32((u64(total_slots) * u64(tid)) / u64(nt));
            u32 hi = u32((u64(total_slots) * u64(tid + 1)) / u64(nt));
            for (u32 i = lo; i < hi; ++i) {
                u8 d = node_keys[i].dom;
                if (d == 255) continue;
                u32 r = parent[i].load(std::memory_order_relaxed);
                u32 b = best[r].load(std::memory_order_relaxed);
                if (b == i) continue;
                const UserKey& alias = node_keys[i];
                const UserKey& canon = node_keys[b];
                if (d == DOM_SHORT) {
                    u64 h = hash_short(alias.lo);
                    abloom[u32(h >> 24) & bloom_mask].fetch_or(bloom_bits(h), std::memory_order_relaxed);
                    u64 cenc;
                    if (__builtin_expect(canon.dom == DOM_SHORT, 1)) {
                        cenc = canon.lo;
                    } else {
                        std::lock_guard<std::mutex> lk(ovf_mu);
                        u64 oid = canon_overflow.size();
                        canon_overflow.push_back(canon);
                        cenc = (1ULL << 63) | oid;
                    }
                    u32 idx = u32(h) & short_mask;
                    for (;;) {
                        u64 exp = 0;
                        if (ashort[idx * 2].compare_exchange_strong(exp, alias.lo,
                                std::memory_order_relaxed, std::memory_order_relaxed)) {
                            short_tab[idx].canon_enc = cenc;
                            break;
                        }
                        idx = (idx + 1u) & short_mask;
                    }
                } else if (d == DOM_HEX32) {
                    u64 h = hash_hex128(alias.hi, alias.lo);
                    abloom[u32(h >> 24) & bloom_mask].fetch_or(bloom_bits(h), std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lk(hex_ex_mu);
                    u32 idx = u32(h) & hex_mask;
                    while (hex_tab[idx].used) idx = (idx + 1u) & hex_mask;
                    hex_tab[idx] = {alias.hi, alias.lo, canon, 1};
                } else {
                    std::lock_guard<std::mutex> lk(hex_ex_mu);
                    exotic_aliases[u32(alias.lo)] = canon;
                }
            }
        });
    }
};

} // namespace hyp
