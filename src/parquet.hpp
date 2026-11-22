#pragma once

#include "common.hpp"
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <string>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

namespace HYP_NS {

// ============================================================================
// Ultra-Fast Zero-Dependency Parquet v1 Writer (Thrift Compact, UNCOMPRESSED)
// - HugeBuf-backed columnar buffers (zero glibc 4KB faults, single capacity check)
// - SIMD 1-bit definition level packer (_mm_movemask_epi8)
// - Raw-pointer RLE_DICTIONARY encoder for person-grouped user_id
// - Zero-copy scatter-gather writev() file output
// ============================================================================

namespace pq {

constexpr i32 TYPE_INT32      = 1;
constexpr i32 TYPE_INT64      = 2;
constexpr i32 TYPE_BYTE_ARRAY = 6;

constexpr i32 CONV_NONE = -1;
constexpr i32 CONV_UTF8 = 0;
constexpr i32 CONV_DATE = 6;

constexpr i32 REP_REQUIRED = 0;
constexpr i32 REP_OPTIONAL = 1;

constexpr i32 ENC_PLAIN            = 0;
constexpr i32 ENC_PLAIN_DICTIONARY = 2;
constexpr i32 ENC_RLE              = 3;

constexpr i32 CODEC_UNCOMPRESSED = 0;

constexpr i32 PAGE_DATA       = 0;
constexpr i32 PAGE_DICTIONARY = 2;

constexpr u8 CT_STOP   = 0;
constexpr u8 CT_I32    = 5;
constexpr u8 CT_I64    = 6;
constexpr u8 CT_BINARY = 8;
constexpr u8 CT_LIST   = 9;
constexpr u8 CT_STRUCT = 12;

class Buf {
public:
    std::vector<u8> data;

    void clear() { data.clear(); }
    void reserve(size_t n) { data.reserve(n); }
    size_t size() const { return data.size(); }

    inline void u8_put(u8 v) { data.push_back(v); }
    inline void put(const void* p, size_t n) {
        if (n == 0) return;
        size_t o = data.size();
        data.resize(o + n);
        std::memcpy(data.data() + o, p, n);
    }
    inline void u32_le(u32 v) { put(&v, 4); }
    inline void i32_le(i32 v) { put(&v, 4); }
    inline void i64_le(i64 v) { put(&v, 8); }

    inline void uvarint(u64 v) {
        u8 tmp[10];
        size_t i = 0;
        while (v >= 0x80u) {
            tmp[i++] = u8((v & 0x7Fu) | 0x80u);
            v >>= 7;
        }
        tmp[i++] = u8(v);
        put(tmp, i);
    }

    static inline u64 zigzag32(i32 v) { return u64((u32(v) << 1) ^ u32(v >> 31)); }
    static inline u64 zigzag64(i64 v) { return (u64(v) << 1) ^ u64(v >> 63); }
};

static inline u8* put_uvarint_raw(u8* p, u64 v) {
    while (v >= 0x80u) {
        *p++ = u8((v & 0x7Fu) | 0x80u);
        v >>= 7;
    }
    *p++ = u8(v);
    return p;
}

class ThriftWriter {
    Buf& b_;
    i16 stack_[16];
    int sp_ = 0;
    i16 last_id_ = 0;

public:
    explicit ThriftWriter(Buf& b) : b_(b) {}

    void field_header(i16 id, u8 ctype) {
        i16 delta = id - last_id_;
        if (delta > 0 && delta <= 15) {
            b_.u8_put(u8((delta << 4) | ctype));
        } else {
            b_.u8_put(ctype);
            b_.uvarint(Buf::zigzag32(id));
        }
        last_id_ = id;
    }

    void write_i32(i16 id, i32 v) {
        field_header(id, CT_I32);
        b_.uvarint(Buf::zigzag32(v));
    }

    void write_i64(i16 id, i64 v) {
        field_header(id, CT_I64);
        b_.uvarint(Buf::zigzag64(v));
    }

    void write_str(i16 id, const std::string& s) {
        field_header(id, CT_BINARY);
        b_.uvarint(s.size());
        b_.put(s.data(), s.size());
    }

    void write_str_elem(const std::string& s) {
        b_.uvarint(s.size());
        b_.put(s.data(), s.size());
    }

    void begin_struct(i16 id) {
        field_header(id, CT_STRUCT);
        stack_[sp_++] = last_id_;
        last_id_ = 0;
    }

    void begin_list_struct() {
        stack_[sp_++] = last_id_;
        last_id_ = 0;
    }

    void end_struct() {
        b_.u8_put(CT_STOP);
        if (sp_ > 0) last_id_ = stack_[--sp_];
    }

    void begin_list(i16 id, u8 elem_type, size_t count) {
        field_header(id, CT_LIST);
        if (count < 15) {
            b_.u8_put(u8((count << 4) | elem_type));
        } else {
            b_.u8_put(u8(0xF0u | elem_type));
            b_.uvarint(count);
        }
    }
};

// Zero-copy scatter-gather FileWriter using POSIX writev()
struct FileWriter {
    int fd = -1;
    i64 total_pos = 0;
    std::vector<iovec> iovs;
    std::vector<u8> inline_pool; // small headers/footers live here so iovs stay valid until finish!

    explicit FileWriter(const std::string& path) {
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) die("open parquet output failed");
        iovs.reserve(64);
        inline_pool.reserve(16384);
    }

    i64 pos() const { return total_pos; }

    void flush_iovs() {
        size_t idx = 0;
        const size_t n = iovs.size();
        while (idx < n) {
            int cnt = int(std::min<size_t>(n - idx, 512));
            ssize_t w = ::writev(fd, &iovs[idx], cnt);
            if (w < 0) {
                if (errno == EINTR) continue;
                die("writev parquet failed");
            }
            size_t rem = size_t(w);
            while (idx < n && rem >= iovs[idx].iov_len) {
                rem -= iovs[idx].iov_len;
                idx++;
            }
            if (idx < n && rem > 0) {
                iovs[idx].iov_base = static_cast<u8*>(iovs[idx].iov_base) + rem;
                iovs[idx].iov_len -= rem;
            }
        }
        iovs.clear();
        inline_pool.clear();
    }

    // Queue an external buffer directly (caller guarantees pointer remains valid until finish/flush!)
    void write_ref(const void* p, size_t n) {
        if (n == 0) return;
        if (iovs.size() >= 480) flush_iovs();
        iovs.push_back({const_cast<void*>(p), n});
        total_pos += i64(n);
    }

    // Copy small ephemeral bytes (headers, magic, footer) into inline_pool
    void write_small(const void* p, size_t n) {
        if (n == 0) return;
        if (inline_pool.size() + n > inline_pool.capacity() || iovs.size() >= 480) {
            flush_iovs();
            if (n > inline_pool.capacity()) inline_pool.reserve(n * 2);
        }
        size_t o = inline_pool.size();
        inline_pool.resize(o + n);
        std::memcpy(inline_pool.data() + o, p, n);
        iovs.push_back({inline_pool.data() + o, n});
        total_pos += i64(n);
    }

    void finish() {
        flush_iovs();
        if (fd >= 0) { ::close(fd); fd = -1; }
    }

    ~FileWriter() {
        if (fd >= 0) { flush_iovs(); ::close(fd); }
    }
};

inline void write_page_header_data(Buf& out, i32 uncomp_size, i32 num_values, i32 encoding) {
    out.clear();
    ThriftWriter tw(out);
    tw.write_i32(1, PAGE_DATA);
    tw.write_i32(2, uncomp_size);
    tw.write_i32(3, uncomp_size);
    tw.begin_struct(5);
    tw.write_i32(1, num_values);
    tw.write_i32(2, encoding);
    tw.write_i32(3, ENC_RLE);
    tw.write_i32(4, ENC_RLE);
    tw.end_struct();
    tw.end_struct();
}

inline void write_page_header_dict(Buf& out, i32 uncomp_size, i32 num_values) {
    out.clear();
    ThriftWriter tw(out);
    tw.write_i32(1, PAGE_DICTIONARY);
    tw.write_i32(2, uncomp_size);
    tw.write_i32(3, uncomp_size);
    tw.begin_struct(7);
    tw.write_i32(1, num_values);
    tw.write_i32(2, ENC_PLAIN);
    tw.end_struct();
    tw.end_struct();
}

struct ColSchema {
    std::string name;
    i32 phys_type;
    i32 conv_type;
    i32 rep_type;
};

struct ColChunkMeta {
    i32 phys_type;
    std::string name;
    std::vector<i32> encodings;
    i64 num_values = 0;
    i64 total_size = 0;
    i64 data_page_offset = 0;
    i64 dict_page_offset = -1;
};

inline void write_footer(FileWriter& fw, const std::vector<ColSchema>& schema,
                         i64 num_rows, const std::vector<ColChunkMeta>& chunks) {
    Buf fb;
    fb.reserve(4096);
    ThriftWriter tw(fb);

    tw.write_i32(1, 1);

    tw.begin_list(2, CT_STRUCT, 1 + schema.size());
    tw.begin_list_struct();
    tw.write_str(4, "schema");
    tw.write_i32(5, i32(schema.size()));
    tw.end_struct();
    for (const auto& c : schema) {
        tw.begin_list_struct();
        tw.write_i32(1, c.phys_type);
        tw.write_i32(3, c.rep_type);
        tw.write_str(4, c.name);
        if (c.conv_type != CONV_NONE) {
            tw.write_i32(6, c.conv_type);
        }
        tw.end_struct();
    }

    tw.write_i64(3, num_rows);

    if (num_rows > 0) {
        tw.begin_list(4, CT_STRUCT, 1);
        tw.begin_list_struct();
        tw.begin_list(1, CT_STRUCT, chunks.size());
        i64 total_byte_size = 0;
        for (const auto& cc : chunks) {
            total_byte_size += cc.total_size;
            tw.begin_list_struct();
            tw.write_i64(2, cc.dict_page_offset >= 0 ? cc.dict_page_offset : cc.data_page_offset);
            tw.begin_struct(3);
            tw.write_i32(1, cc.phys_type);
            tw.begin_list(2, CT_I32, cc.encodings.size());
            for (i32 enc : cc.encodings) fb.uvarint(Buf::zigzag32(enc));
            tw.begin_list(3, CT_BINARY, 1);
            tw.write_str_elem(cc.name);
            tw.write_i32(4, CODEC_UNCOMPRESSED);
            tw.write_i64(5, cc.num_values);
            tw.write_i64(6, cc.total_size);
            tw.write_i64(7, cc.total_size);
            tw.write_i64(9, cc.data_page_offset);
            if (cc.dict_page_offset >= 0) {
                tw.write_i64(11, cc.dict_page_offset);
            }
            tw.end_struct();
            tw.end_struct();
        }
        tw.write_i64(2, total_byte_size);
        tw.write_i64(3, num_rows);
        tw.end_struct();
    } else {
        tw.begin_list(4, CT_STRUCT, 0);
    }

    tw.write_str(6, "hyperion");
    tw.end_struct();

    u32 footer_len = u32(fb.size());
    fw.write_small(fb.data.data(), fb.size());
    fw.write_small(&footer_len, 4);
    fw.write_small("PAR1", 4);
}

// SIMD-accelerated 1-bit definition level packer (16 bytes -> 2 packed bytes per _mm_movemask_epi8!)
inline u8* encode_def_levels_1bit_raw(const u8* defs, size_t n, u8* out) {
    if (n == 0) {
        u32 z = 0;
        std::memcpy(out, &z, 4);
        return out + 4;
    }
    u8* len_ptr = out;
    u8* p = out + 4;
    size_t i = 0;
    while (i < n) {
        size_t rem = n - i;
        size_t groups = (rem + 7) / 8;
        if (groups > 63) groups = 63;
        *p++ = u8((groups << 1) | 1u); // groups <= 63 => fits in 1 byte varint!
        size_t g = 0;
        // Fast SIMD path: 2 groups (16 bytes) at a time while >= 16 bytes remain!
        while (g + 2 <= groups && i + (g + 2) * 8 <= n) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(defs + i + g * 8));
            u16 mask = u16(_mm_movemask_epi8(_mm_slli_epi16(v, 7)));
            std::memcpy(p, &mask, 2);
            p += 2;
            g += 2;
        }
        for (; g < groups; ++g) {
            u8 byte = 0;
            size_t base = i + g * 8;
            size_t lim = std::min<size_t>(8, n - base);
            for (size_t b = 0; b < lim; ++b) {
                byte |= u8((defs[base + b] & 1u) << b);
            }
            *p++ = byte;
        }
        i += groups * 8;
    }
    u32 rle_len = u32(p - (out + 4));
    std::memcpy(len_ptr, &rle_len, 4);
    return p;
}

inline u8* encode_dict_indices_small_raw(const u16* vals, size_t n, u32 bit_width, u8* p) {
    *p++ = u8(bit_width);
    if (n == 0) return p;
    if (bit_width == 0) {
        return put_uvarint_raw(p, u64(n) << 1);
    }
    size_t i = 0;
    while (i < n) {
        size_t rem = n - i;
        size_t groups = (rem + 7) / 8;
        if (groups > 63) groups = 63;
        *p++ = u8((groups << 1) | 1u);
        for (size_t g = 0; g < groups; ++g) {
            u64 acc = 0;
            size_t base = i + g * 8;
            size_t lim = std::min<size_t>(8, n - base);
            for (size_t b = 0; b < lim; ++b) {
                acc |= u64(vals[base + b]) << (b * bit_width);
            }
            std::memcpy(p, &acc, bit_width);
            p += bit_width;
        }
        i += groups * 8;
    }
    return p;
}

// HugeBuf-backed per-thread user_daily columns (single capacity check, zero glibc 4KB faults!)
struct UserDailyCols {
    HugeBuf<u8>  user_dict_bytes;
    HugeBuf<u32> user_run_lens;

    i32* day      = nullptr;
    i32* sessions = nullptr;
    i32* events   = nullptr;
    i32* rev32    = nullptr;
    i64* rev64    = nullptr;
    u8*  cat_def  = nullptr;
    size_t nrows  = 0;
    size_t rcap   = 0;
    bool use_rev64 = false;

    HugeBuf<u16> cat_idx;

    void reserve(size_t est_rows, size_t est_people) {
        user_dict_bytes.init_reserve(est_people * 20);
        user_run_lens.init_reserve(est_people);
        cat_idx.init_reserve(est_rows / 4 + 1024);
        if (!day) {
            rcap = est_rows < 4096 ? 4096 : est_rows;
            day      = static_cast<i32*>(huge_alloc(rcap * sizeof(i32)));
            sessions = static_cast<i32*>(huge_alloc(rcap * sizeof(i32)));
            events   = static_cast<i32*>(huge_alloc(rcap * sizeof(i32)));
            rev32    = static_cast<i32*>(huge_alloc(rcap * sizeof(i32)));
            cat_def  = static_cast<u8*>(huge_alloc(rcap * sizeof(u8)));
        }
    }

    void grow_rows() {
        size_t ncap = rcap ? (rcap << 1) : 4096;
        auto grow = [&](auto*& p) {
            using T = std::remove_pointer_t<std::remove_reference_t<decltype(p)>>;
            T* np = static_cast<T*>(huge_alloc(ncap * sizeof(T)));
            if (nrows > 0 && p) std::memcpy(np, p, nrows * sizeof(T));
            p = np;
        };
        grow(day);
        grow(sessions);
        grow(events);
        if (!use_rev64) grow(rev32);
        else grow(rev64);
        grow(cat_def);
        rcap = ncap;
    }

    void promote_rev64() {
        rev64 = static_cast<i64*>(huge_alloc(rcap * sizeof(i64)));
        for (size_t i = 0; i < nrows; ++i) rev64[i] = i64(rev32[i]);
        use_rev64 = true;
    }

    inline void add_person_name(const u8* p, u32 len, u32 active_days) {
        user_dict_bytes.ensure_extra(4u + len);
        u8* dst = user_dict_bytes.ptr + user_dict_bytes.len;
        std::memcpy(dst, &len, 4);
        std::memcpy(dst + 4, p, len);
        user_dict_bytes.len += 4u + len;
        user_run_lens.push(active_days);
    }

    inline void add_row(i32 d, i32 s, i32 e, i64 rev, i32 top_cat) {
        if (__builtin_expect(nrows == rcap, 0)) grow_rows();
        const size_t i = nrows++;
        day[i]      = d;
        sessions[i] = s;
        events[i]   = e;
        if (__builtin_expect(!use_rev64, 1)) {
            if (__builtin_expect(rev <= INT32_MAX, 1)) {
                rev32[i] = i32(rev);
            } else {
                promote_rev64();
                rev64[i] = rev;
            }
        } else {
            rev64[i] = rev;
        }
        if (top_cat >= 0) {
            cat_def[i] = 1;
            cat_idx.push(u16(top_cat));
        } else {
            cat_def[i] = 0;
        }
    }

    void write_parquet(const std::string& path, const std::vector<std::string>& categories) const {
        FileWriter fw(path);
        fw.write_small("PAR1", 4);

        const i64 nr = i64(nrows);
        const bool rev_i32 = !use_rev64;
        std::vector<ColSchema> schema = {
            {"user_id",       TYPE_BYTE_ARRAY, CONV_UTF8, REP_REQUIRED},
            {"day",           TYPE_INT32,      CONV_DATE, REP_REQUIRED},
            {"sessions",      TYPE_INT32,      CONV_NONE, REP_REQUIRED},
            {"events",        TYPE_INT32,      CONV_NONE, REP_REQUIRED},
            {"revenue_cents", rev_i32 ? TYPE_INT32 : TYPE_INT64, CONV_NONE, REP_REQUIRED},
            {"top_category",  TYPE_BYTE_ARRAY, CONV_UTF8, REP_OPTIONAL},
        };

        std::vector<ColChunkMeta> metas;
        // Keep encoded buffers alive until fw.finish()!
        std::vector<u8> uid_rle_buf;
        std::vector<u8> cat_dict_buf;
        std::vector<u8> cat_data_buf;

        if (nr > 0) {
            metas.reserve(6);
            Buf hdr;

            // 1. user_id: RLE_DICTIONARY (1 dict entry + 1 RLE run per person, raw pointer loop!)
            {
                ColChunkMeta cm;
                cm.phys_type = TYPE_BYTE_ARRAY;
                cm.name = "user_id";
                cm.encodings = {ENC_PLAIN, ENC_PLAIN_DICTIONARY, ENC_RLE};
                cm.num_values = nr;
                cm.dict_page_offset = fw.pos();

                const u32 ndict = u32(user_run_lens.size());
                write_page_header_dict(hdr, i32(user_dict_bytes.size()), i32(ndict));
                fw.write_small(hdr.data.data(), hdr.size());
                fw.write_ref(user_dict_bytes.data(), user_dict_bytes.size());

                cm.data_page_offset = fw.pos();
                u32 bw = (ndict <= 1) ? 0u : (32u - u32(__builtin_clz(ndict - 1u)));
                u32 val_bytes = (bw + 7u) >> 3;
                uid_rle_buf.resize(16 + size_t(ndict) * (val_bytes + 5));
                u8* p = uid_rle_buf.data();
                *p++ = u8(bw);
                if (bw == 0) {
                    p = put_uvarint_raw(p, u64(nr) << 1);
                } else {
                    const u32* rl = user_run_lens.data();
                    if (val_bytes == 3) {
                        for (u32 i = 0; i < ndict; ++i) {
                            u32 r = rl[i];
                            if (__builtin_expect(r < 64u, 1)) *p++ = u8(r << 1);
                            else p = put_uvarint_raw(p, u64(r) << 1);
                            p[0] = u8(i); p[1] = u8(i >> 8); p[2] = u8(i >> 16);
                            p += 3;
                        }
                    } else if (val_bytes == 2) {
                        for (u32 i = 0; i < ndict; ++i) {
                            u32 r = rl[i];
                            if (__builtin_expect(r < 64u, 1)) *p++ = u8(r << 1);
                            else p = put_uvarint_raw(p, u64(r) << 1);
                            p[0] = u8(i); p[1] = u8(i >> 8);
                            p += 2;
                        }
                    } else {
                        for (u32 i = 0; i < ndict; ++i) {
                            p = put_uvarint_raw(p, u64(rl[i]) << 1);
                            std::memcpy(p, &i, val_bytes);
                            p += val_bytes;
                        }
                    }
                }
                size_t rle_sz = size_t(p - uid_rle_buf.data());
                write_page_header_data(hdr, i32(rle_sz), i32(nr), ENC_PLAIN_DICTIONARY);
                fw.write_small(hdr.data.data(), hdr.size());
                fw.write_ref(uid_rle_buf.data(), rle_sz);
                cm.total_size = fw.pos() - cm.dict_page_offset;
                metas.push_back(std::move(cm));
            }

            // 2..4. day, sessions, events (INT32 PLAIN, zero-copy writev!)
            auto write_i32_col = [&](const char* name, const i32* ptr) {
                ColChunkMeta cm;
                cm.phys_type = TYPE_INT32;
                cm.name = name;
                cm.encodings = {ENC_PLAIN, ENC_RLE};
                cm.num_values = nr;
                cm.data_page_offset = fw.pos();
                i32 sz = i32(nrows * 4);
                write_page_header_data(hdr, sz, i32(nr), ENC_PLAIN);
                fw.write_small(hdr.data.data(), hdr.size());
                fw.write_ref(ptr, size_t(sz));
                cm.total_size = fw.pos() - cm.data_page_offset;
                metas.push_back(std::move(cm));
            };
            write_i32_col("day", day);
            write_i32_col("sessions", sessions);
            write_i32_col("events", events);

            // 5. revenue_cents (INT32 if fits else INT64, PLAIN, zero-copy writev!)
            {
                ColChunkMeta cm;
                cm.name = "revenue_cents";
                cm.encodings = {ENC_PLAIN, ENC_RLE};
                cm.num_values = nr;
                cm.data_page_offset = fw.pos();
                if (rev_i32) {
                    cm.phys_type = TYPE_INT32;
                    i32 sz = i32(nrows * 4);
                    write_page_header_data(hdr, sz, i32(nr), ENC_PLAIN);
                    fw.write_small(hdr.data.data(), hdr.size());
                    fw.write_ref(rev32, size_t(sz));
                } else {
                    cm.phys_type = TYPE_INT64;
                    i32 sz = i32(nrows * 8);
                    write_page_header_data(hdr, sz, i32(nr), ENC_PLAIN);
                    fw.write_small(hdr.data.data(), hdr.size());
                    fw.write_ref(rev64, size_t(sz));
                }
                cm.total_size = fw.pos() - cm.data_page_offset;
                metas.push_back(std::move(cm));
            }

            // 6. top_category (OPTIONAL BYTE_ARRAY UTF8, RLE_DICTIONARY, SIMD 1-bit def levels!)
            {
                ColChunkMeta cm;
                cm.phys_type = TYPE_BYTE_ARRAY;
                cm.name = "top_category";
                cm.encodings = {ENC_PLAIN, ENC_PLAIN_DICTIONARY, ENC_RLE};
                cm.num_values = nr;
                cm.dict_page_offset = fw.pos();

                size_t dsz = 0;
                for (const auto& cat : categories) dsz += 4 + cat.size();
                cat_dict_buf.resize(dsz);
                u8* dp = cat_dict_buf.data();
                for (const auto& cat : categories) {
                    u32 len = u32(cat.size());
                    std::memcpy(dp, &len, 4);
                    std::memcpy(dp + 4, cat.data(), len);
                    dp += 4 + len;
                }
                write_page_header_dict(hdr, i32(dsz), i32(categories.size()));
                fw.write_small(hdr.data.data(), hdr.size());
                fw.write_ref(cat_dict_buf.data(), dsz);

                cm.data_page_offset = fw.pos();
                u32 ncats = u32(categories.size());
                u32 bw = (ncats <= 1) ? 0u : (32u - u32(__builtin_clz(ncats - 1u)));
                size_t max_cat_bytes = 64 + (nrows / 8) * 2 + (cat_idx.size() * 4);
                cat_data_buf.resize(max_cat_bytes);
                u8* p = encode_def_levels_1bit_raw(cat_def, nrows, cat_data_buf.data());
                if (bw <= 8) {
                    p = encode_dict_indices_small_raw(cat_idx.data(), cat_idx.size(), bw, p);
                } else {
                    u32 vb = (bw + 7u) >> 3;
                    *p++ = u8(bw);
                    for (size_t i = 0; i < cat_idx.size(); ++i) {
                        p = put_uvarint_raw(p, 2u);
                        u16 v = cat_idx.data()[i];
                        std::memcpy(p, &v, vb);
                        p += vb;
                    }
                }
                size_t csz = size_t(p - cat_data_buf.data());
                write_page_header_data(hdr, i32(csz), i32(nr), ENC_PLAIN_DICTIONARY);
                fw.write_small(hdr.data.data(), hdr.size());
                fw.write_ref(cat_data_buf.data(), csz);
                cm.total_size = fw.pos() - cm.dict_page_offset;
                metas.push_back(std::move(cm));
            }
        }

        write_footer(fw, schema, nr, metas);
        fw.finish();
    }
};

struct CampaignDailyRow {
    u32 camp_idx;
    i32 day;
    i32 sessions;
    i32 purchasing_sessions;
    i64 revenue_cents;
    i32 assisted_sessions;
};

inline void write_campaign_daily_parquet(const std::string& path,
                                         const std::vector<CampaignDailyRow>& rows,
                                         const std::vector<std::string>& camp_ids,
                                         const std::vector<std::string>& camp_channels) {
    FileWriter fw(path);
    fw.write_small("PAR1", 4);

    const i64 nrows = i64(rows.size());
    std::vector<ColSchema> schema = {
        {"campaign_id",         TYPE_BYTE_ARRAY, CONV_UTF8, REP_REQUIRED},
        {"channel",             TYPE_BYTE_ARRAY, CONV_UTF8, REP_REQUIRED},
        {"day",                 TYPE_INT32,      CONV_DATE, REP_REQUIRED},
        {"sessions",            TYPE_INT32,      CONV_NONE, REP_REQUIRED},
        {"purchasing_sessions", TYPE_INT32,      CONV_NONE, REP_REQUIRED},
        {"revenue_cents",       TYPE_INT64,      CONV_NONE, REP_REQUIRED},
        {"assisted_sessions",   TYPE_INT32,      CONV_NONE, REP_REQUIRED},
    };

    std::vector<ColChunkMeta> metas;
    Buf cid_buf, ch_buf;
    std::vector<i32> col_day, col_sess, col_ps, col_as;
    std::vector<i64> col_rev;

    if (nrows > 0) {
        metas.reserve(7);
        Buf hdr;

        auto write_str_col = [&](const char* name, const std::vector<std::string>& dict, Buf& dst) {
            ColChunkMeta cm;
            cm.phys_type = TYPE_BYTE_ARRAY;
            cm.name = name;
            cm.encodings = {ENC_PLAIN, ENC_RLE};
            cm.num_values = nrows;
            cm.data_page_offset = fw.pos();
            dst.clear();
            dst.reserve(size_t(nrows) * 12);
            for (const auto& r : rows) {
                const std::string& s = dict[r.camp_idx];
                u32 len = u32(s.size());
                dst.u32_le(len);
                dst.put(s.data(), len);
            }
            write_page_header_data(hdr, i32(dst.size()), i32(nrows), ENC_PLAIN);
            fw.write_small(hdr.data.data(), hdr.size());
            fw.write_ref(dst.data.data(), dst.size());
            cm.total_size = fw.pos() - cm.data_page_offset;
            metas.push_back(std::move(cm));
        };

        auto write_i32_col = [&](const char* name, std::vector<i32>& vals, auto getter) {
            ColChunkMeta cm;
            cm.phys_type = TYPE_INT32;
            cm.name = name;
            cm.encodings = {ENC_PLAIN, ENC_RLE};
            cm.num_values = nrows;
            cm.data_page_offset = fw.pos();
            vals.resize(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) vals[i] = getter(rows[i]);
            i32 sz = i32(vals.size() * 4);
            write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
            fw.write_small(hdr.data.data(), hdr.size());
            fw.write_ref(vals.data(), size_t(sz));
            cm.total_size = fw.pos() - cm.data_page_offset;
            metas.push_back(std::move(cm));
        };

        write_str_col("campaign_id", camp_ids, cid_buf);
        write_str_col("channel", camp_channels, ch_buf);
        write_i32_col("day", col_day, [](const CampaignDailyRow& r) { return r.day; });
        write_i32_col("sessions", col_sess, [](const CampaignDailyRow& r) { return r.sessions; });
        write_i32_col("purchasing_sessions", col_ps, [](const CampaignDailyRow& r) { return r.purchasing_sessions; });

        {
            ColChunkMeta cm;
            cm.phys_type = TYPE_INT64;
            cm.name = "revenue_cents";
            cm.encodings = {ENC_PLAIN, ENC_RLE};
            cm.num_values = nrows;
            cm.data_page_offset = fw.pos();
            col_rev.resize(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) col_rev[i] = rows[i].revenue_cents;
            i32 sz = i32(col_rev.size() * 8);
            write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
            fw.write_small(hdr.data.data(), hdr.size());
            fw.write_ref(col_rev.data(), size_t(sz));
            cm.total_size = fw.pos() - cm.data_page_offset;
            metas.push_back(std::move(cm));
        }

        write_i32_col("assisted_sessions", col_as, [](const CampaignDailyRow& r) { return r.assisted_sessions; });
    }

    write_footer(fw, schema, nrows, metas);
    fw.finish();
}

} // namespace pq
} // namespace HYP_NS
