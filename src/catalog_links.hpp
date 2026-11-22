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

// List all *.csv files in a directory (sorted for determinism)
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

// Parse manifest.json: {"window_start": "...", "window_end": "..."}
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
    // Categories sorted byte-wise ascending (including "unknown"), so smaller id == byte-wise smaller!
    std::vector<std::string> categories;
    u16 unknown_cat = 0;

    static constexpr u32 PROD_FAST_MAX = 65536;
    u16 prod_fast[PROD_FAST_MAX];
    std::unordered_map<std::string, u16> prod_slow;
    bool has_slow_prods = false;

    std::vector<std::string> camp_ids;
    std::vector<std::string> camp_channels;
    u32 num_campaigns = 0; // organic == num_campaigns

    static constexpr u32 CAMP_FAST_MAX = 4096;
    i32 camp_fast[CAMP_FAST_MAX];
    std::unordered_map<std::string, i32> camp_slow;
    bool has_slow_camps = false;

    static inline bool parse_prefixed_uint(const u8* p, size_t n, u32 prefix_len, u32 max_v, u32& out) {
        if (n <= prefix_len || n > prefix_len + 6) return false;
        const u8* d = p + prefix_len;
        size_t dn = n - prefix_len;
        if (d[0] < '1' || d[0] > '9') return false; // no leading zeros so canonical
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

        // Collect unique categories + "unknown" and sort byte-wise
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
        // Add "organic" as sentinel at index num_campaigns
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
// Links Resolver: Parallel Parse + DSU + 1-Word Blocked Bloom + Compact Alias Table
// ============================================================================

struct LinkEdge {
    UserKey a;
    UserKey b;
};

struct ShortAliasSlot {
    u64 key;        // 0 = empty (DOM_SHORT keys are always non-zero!)
    UserKey canon;
};

struct HexAliasSlot {
    u64 hi;
    u64 lo;
    UserKey canon;
    u8 used;
};

struct LinkResolver {
    bool has_aliases = false;

    // 1-word register-blocked Bloom filter (3 bits inside a single u64 word -> 1 load!)
    u64* bloom = nullptr;
    u32 bloom_mask = 0;

    // Compact read-only open-addressing tables for NON-CANONICAL aliases only
    ShortAliasSlot* short_tab = nullptr;
    u32 short_mask = 0;

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
            if (s.key == key) return s.canon;
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

    inline UserKey resolve(const UserKey& k) const {
        if (__builtin_expect(!has_aliases, 0)) return k;
        if (__builtin_expect(k.dom == DOM_SHORT, 1)) {
            u64 h = hash_short(k.lo);
            if (!bloom_may_contain(h)) return k;
            return resolve_short_hit(k.lo, h);
        }
        if (k.dom == DOM_HEX32) {
            u64 h = hash_hex128(k.hi, k.lo);
            if (!bloom_may_contain(h)) return k;
            return resolve_hex_hit(k.hi, k.lo, h);
        }
        auto it = exotic_aliases.find(u32(k.lo));
        return it != exotic_aliases.end() ? it->second : k;
    }

    void build(const std::vector<MFile>& link_files, ThreadPool& pool, ExoticInterner& ex) {
        int nt = pool.size();
        // Build 1 MiB byte-range chunks across all link files
        struct LChunk { u32 file; size_t s, e; };
        std::vector<LChunk> chunks;
        for (u32 fi = 0; fi < link_files.size(); ++fi) {
            const MFile& mf = link_files[fi];
            if (mf.size == 0) continue;
            const u8* nl = static_cast<const u8*>(std::memchr(mf.data, '\n', mf.size));
            if (!nl) continue; // header only
            size_t body = size_t(nl - mf.data) + 1;
            if (body >= mf.size) continue;
            constexpr size_t CSZ = 1u << 20;
            for (size_t off = body; off < mf.size; off += CSZ) {
                chunks.push_back({fi, off, std::min(off + CSZ, mf.size)});
            }
        }
        if (chunks.empty()) {
            has_aliases = false;
            return;
        }

        std::vector<std::vector<LinkEdge>> t_edges(nt);
        std::atomic<size_t> next_ch{0};
        pool.run([&](int tid) {
            auto& out = t_edges[tid];
            out.reserve(65536);
            for (;;) {
                size_t ci = next_ch.fetch_add(1, std::memory_order_relaxed);
                if (ci >= chunks.size()) break;
                const auto& ch = chunks[ci];
                const MFile& mf = link_files[ch.file];
                size_t s = ch.s;
                // Align to next line start if not at file body start
                if (s > 0 && mf.data[s - 1] != '\n') {
                    const u8* nl = static_cast<const u8*>(std::memchr(mf.data + s, '\n', mf.size - s));
                    if (!nl) continue;
                    s = size_t(nl - mf.data) + 1;
                }
                if (s >= ch.e) continue;
                // Extend chunk end to include the line starting before ch.e
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

        // Open-addressing hash table assigning dense node IDs 0..N-1 to unique linked UserKeys
        // Using flat table sized to >= 4 * total_edges (load factor <= 0.5)
        u32 cap = 1024;
        while (cap < total_edges * 4 && cap < (1u << 28)) cap <<= 1;
        u32 mask = cap - 1;

        struct NodeSlot {
            UserKey key;
            u32 id;
            u8 used;
        };
        NodeSlot* ntab = static_cast<NodeSlot*>(huge_alloc(size_t(cap) * sizeof(NodeSlot)));
        std::vector<UserKey> keys;
        keys.reserve(total_edges * 2);
        std::vector<u32> parent;
        parent.reserve(total_edges * 2);

        auto get_or_add = [&](const UserKey& k) -> u32 {
            u64 h = hash_user_key(k);
            u32 idx = u32(h) & mask;
            for (;;) {
                NodeSlot& s = ntab[idx];
                if (!s.used) {
                    u32 id = u32(keys.size());
                    s.key = k;
                    s.id = id;
                    s.used = 1;
                    keys.push_back(k);
                    parent.push_back(id);
                    return id;
                }
                if (s.key == k) return s.id;
                idx = (idx + 1u) & mask;
            }
        };

        auto find_root = [&](u32 x) -> u32 {
            u32 r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != r) {
                u32 nx = parent[x];
                parent[x] = r;
                x = nx;
            }
            return r;
        };

        // Union always attaches larger name under byte-wise smaller root!
        // Thus after all unions, root IS the byte-wise smallest name of its component!
        for (int t = 0; t < nt; ++t) {
            const auto& edges = t_edges[t];
            const size_t en = edges.size();
            for (size_t i = 0; i < en; ++i) {
                if (i + 12 < en) {
                    __builtin_prefetch(&ntab[u32(hash_user_key(edges[i + 12].a)) & mask], 1, 1);
                    __builtin_prefetch(&ntab[u32(hash_user_key(edges[i + 12].b)) & mask], 1, 1);
                }
                u32 ia = get_or_add(edges[i].a);
                u32 ib = get_or_add(edges[i].b);
                u32 ra = find_root(ia);
                u32 rb = find_root(ib);
                if (ra != rb) {
                    if (user_key_less(keys[ra], keys[rb], ex)) {
                        parent[rb] = ra;
                    } else {
                        parent[ra] = rb;
                    }
                }
            }
        }

        const u32 nnodes = u32(keys.size());
        // Compress all paths and count non-canonical aliases by domain
        u32 n_short = 0, n_hex = 0, n_exotic = 0;
        for (u32 i = 0; i < nnodes; ++i) {
            u32 r = find_root(i);
            if (r != i) {
                u8 d = keys[i].dom;
                if (d == DOM_SHORT) ++n_short;
                else if (d == DOM_HEX32) ++n_hex;
                else ++n_exotic;
            }
        }

        u32 total_aliases = n_short + n_hex + n_exotic;
        if (total_aliases == 0) {
            has_aliases = false;
            return;
        }
        has_aliases = true;

        // Allocate 1-word register-blocked Bloom filter (~20 bits per alias, power of 2 words)
        u32 bwords = 4096;
        while (u64(bwords) * 64ULL < u64(total_aliases) * 20ULL) bwords <<= 1;
        bloom_mask = bwords - 1;
        bloom = static_cast<u64*>(huge_alloc(size_t(bwords) * sizeof(u64)));

        if (n_short > 0) {
            u32 scap = 64;
            while (scap < n_short * 3) scap <<= 1;
            short_mask = scap - 1;
            short_tab = static_cast<ShortAliasSlot*>(huge_alloc(size_t(scap) * sizeof(ShortAliasSlot)));
        }
        if (n_hex > 0) {
            u32 hcap = 64;
            while (hcap < n_hex * 3) hcap <<= 1;
            hex_mask = hcap - 1;
            hex_tab = static_cast<HexAliasSlot*>(huge_alloc(size_t(hcap) * sizeof(HexAliasSlot)));
        }

        // Populate Bloom + compact read-only non-canonical alias tables
        for (u32 i = 0; i < nnodes; ++i) {
            u32 r = parent[i];
            if (r == i) continue; // canonical root: skip!
            const UserKey& alias = keys[i];
            const UserKey& canon = keys[r];
            if (alias.dom == DOM_SHORT) {
                u64 h = hash_short(alias.lo);
                bloom[u32(h >> 24) & bloom_mask] |= bloom_bits(h);
                u32 idx = u32(h) & short_mask;
                while (short_tab[idx].key != 0) idx = (idx + 1u) & short_mask;
                short_tab[idx] = {alias.lo, canon};
            } else if (alias.dom == DOM_HEX32) {
                u64 h = hash_hex128(alias.hi, alias.lo);
                bloom[u32(h >> 24) & bloom_mask] |= bloom_bits(h);
                u32 idx = u32(h) & hex_mask;
                while (hex_tab[idx].used) idx = (idx + 1u) & hex_mask;
                hex_tab[idx] = {alias.hi, alias.lo, canon, 1};
            } else {
                exotic_aliases[u32(alias.lo)] = canon;
            }
        }
    }
};

} // namespace hyp
