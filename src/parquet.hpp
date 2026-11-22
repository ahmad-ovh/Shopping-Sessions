#pragma once

#include "common.hpp"
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace hyp {

// ============================================================================
// Minimal Zero-Dependency Parquet v1 Writer (Thrift Compact Protocol, UNCOMPRESSED)
// Supports:
//   - INT32 / DATE (PLAIN)
//   - INT64 (PLAIN)
//   - BYTE_ARRAY / UTF8 (PLAIN)
//   - BYTE_ARRAY / UTF8 REQUIRED with RLE_DICTIONARY (ideal for person-grouped user_id!)
//   - BYTE_ARRAY / UTF8 OPTIONAL with RLE_DICTIONARY + 1-bit bit-packed def levels (top_category!)
// ============================================================================

namespace pq {

// Parquet Physical Types
constexpr i32 TYPE_INT32      = 1;
constexpr i32 TYPE_INT64      = 2;
constexpr i32 TYPE_BYTE_ARRAY = 6;

// ConvertedType
constexpr i32 CONV_NONE = -1;
constexpr i32 CONV_UTF8 = 0;
constexpr i32 CONV_DATE = 6;

// FieldRepetitionType
constexpr i32 REP_REQUIRED = 0;
constexpr i32 REP_OPTIONAL = 1;

// Encoding
constexpr i32 ENC_PLAIN            = 0;
constexpr i32 ENC_PLAIN_DICTIONARY = 2;
constexpr i32 ENC_RLE              = 3;

// CompressionCodec
constexpr i32 CODEC_UNCOMPRESSED = 0;

// PageType
constexpr i32 PAGE_DATA       = 0;
constexpr i32 PAGE_DICTIONARY = 2;

// Thrift Compact Protocol types
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

class ThriftWriter {
    Buf& b_;
    std::vector<i16> stack_;
    i16 last_id_ = 0;

public:
    explicit ThriftWriter(Buf& b) : b_(b) { stack_.reserve(16); }

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
        stack_.push_back(last_id_);
        last_id_ = 0;
    }

    void begin_list_struct() {
        stack_.push_back(last_id_);
        last_id_ = 0;
    }

    void end_struct() {
        b_.u8_put(CT_STOP);
        if (!stack_.empty()) {
            last_id_ = stack_.back();
            stack_.pop_back();
        }
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

struct FileWriter {
    int fd = -1;
    std::vector<u8> buf;
    i64 written = 0;

    explicit FileWriter(const std::string& path, size_t cap = 1u << 20) {
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) die("open parquet output failed");
        buf.reserve(cap);
    }

    i64 pos() const { return written + i64(buf.size()); }

    void flush() {
        const u8* p = buf.data();
        size_t rem = buf.size();
        while (rem > 0) {
            ssize_t w = ::write(fd, p, rem);
            if (w < 0) {
                if (errno == EINTR) continue;
                die("write parquet failed");
            }
            p += w;
            rem -= size_t(w);
            written += w;
        }
        buf.clear();
    }

    void write_bytes(const void* p, size_t n) {
        if (n == 0) return;
        if (buf.size() + n > buf.capacity()) {
            flush();
            if (n >= buf.capacity()) {
                const u8* q = static_cast<const u8*>(p);
                size_t rem = n;
                while (rem > 0) {
                    ssize_t w = ::write(fd, q, rem);
                    if (w < 0) {
                        if (errno == EINTR) continue;
                        die("write parquet large failed");
                    }
                    q += w;
                    rem -= size_t(w);
                    written += w;
                }
                return;
            }
        }
        size_t o = buf.size();
        buf.resize(o + n);
        std::memcpy(buf.data() + o, p, n);
    }

    void finish() {
        flush();
        if (fd >= 0) { ::close(fd); fd = -1; }
    }

    ~FileWriter() {
        if (fd >= 0) { flush(); ::close(fd); }
    }
};

inline void write_page_header_data(Buf& out, i32 uncomp_size, i32 num_values, i32 encoding) {
    out.clear();
    ThriftWriter tw(out);
    tw.write_i32(1, PAGE_DATA);
    tw.write_i32(2, uncomp_size);
    tw.write_i32(3, uncomp_size);
    tw.begin_struct(5); // DataPageHeader
    tw.write_i32(1, num_values);
    tw.write_i32(2, encoding);
    tw.write_i32(3, ENC_RLE); // definition_level_encoding
    tw.write_i32(4, ENC_RLE); // repetition_level_encoding
    tw.end_struct();
    tw.end_struct();
}

inline void write_page_header_dict(Buf& out, i32 uncomp_size, i32 num_values) {
    out.clear();
    ThriftWriter tw(out);
    tw.write_i32(1, PAGE_DICTIONARY);
    tw.write_i32(2, uncomp_size);
    tw.write_i32(3, uncomp_size);
    tw.begin_struct(7); // DictionaryPageHeader
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
    i64 dict_page_offset = -1; // -1 if no dictionary page
};

inline void write_footer(FileWriter& fw, const std::vector<ColSchema>& schema,
                         i64 num_rows, const std::vector<ColChunkMeta>& chunks) {
    Buf fb;
    fb.reserve(4096);
    ThriftWriter tw(fb);

    tw.write_i32(1, 1); // version = 1

    // field 2: schema (list<SchemaElement>)
    tw.begin_list(2, CT_STRUCT, 1 + schema.size());
    // Root element
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

    // field 4: row_groups
    if (num_rows > 0) {
        tw.begin_list(4, CT_STRUCT, 1);
        tw.begin_list_struct();
        // field 1: columns
        tw.begin_list(1, CT_STRUCT, chunks.size());
        i64 total_byte_size = 0;
        for (const auto& cc : chunks) {
            total_byte_size += cc.total_size;
            tw.begin_list_struct();
            tw.write_i64(2, cc.dict_page_offset >= 0 ? cc.dict_page_offset : cc.data_page_offset);
            tw.begin_struct(3); // ColumnMetaData
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
            tw.end_struct(); // ColumnMetaData
            tw.end_struct(); // ColumnChunk
        }
        tw.write_i64(2, total_byte_size);
        tw.write_i64(3, num_rows);
        tw.end_struct(); // RowGroup
    } else {
        tw.begin_list(4, CT_STRUCT, 0);
    }

    tw.write_str(6, "hyperion");
    tw.end_struct();

    u32 footer_len = u32(fb.size());
    fw.write_bytes(fb.data.data(), fb.size());
    fw.write_bytes(&footer_len, 4);
    fw.write_bytes("PAR1", 4);
}

// Helper: encode 1-bit definition levels using Parquet RLE/bit-packed hybrid
// (bit-packed groups of 8 values = exact 1 bit/row + tiny header!)
inline void encode_def_levels_1bit(const std::vector<u8>& defs, Buf& out) {
    out.clear();
    const size_t n = defs.size();
    if (n == 0) {
        out.u32_le(0);
        return;
    }
    Buf rle;
    rle.reserve((n + 7) / 8 + 32);
    size_t i = 0;
    while (i < n) {
        // Pack up to 63*8 = 504 groups per bit-packed header
        size_t rem = n - i;
        size_t groups = (rem + 7) / 8;
        if (groups > 63) groups = 63;
        rle.uvarint((groups << 1) | 1u);
        for (size_t g = 0; g < groups; ++g) {
            u8 byte = 0;
            size_t base = i + g * 8;
            size_t lim = std::min<size_t>(8, n - base);
            for (size_t b = 0; b < lim; ++b) {
                byte |= u8((defs[base + b] & 1u) << b);
            }
            rle.u8_put(byte);
        }
        i += groups * 8;
    }
    out.u32_le(u32(rle.size()));
    out.put(rle.data.data(), rle.size());
}

// Helper: encode u8/u16 indices with given bit_width (1..8) using bit-packed groups of 8
inline void encode_dict_indices_small(const std::vector<u16>& vals, u32 bit_width, Buf& out) {
    out.u8_put(u8(bit_width));
    const size_t n = vals.size();
    if (n == 0) return;
    if (bit_width == 0) {
        // All indices are 0: single RLE run of length n with 0 value bytes
        out.uvarint(u64(n) << 1);
        return;
    }
    size_t i = 0;
    while (i < n) {
        size_t rem = n - i;
        size_t groups = (rem + 7) / 8;
        if (groups > 63) groups = 63;
        out.uvarint((groups << 1) | 1u);
        for (size_t g = 0; g < groups; ++g) {
            u64 acc = 0;
            size_t base = i + g * 8;
            size_t lim = std::min<size_t>(8, n - base);
            for (size_t b = 0; b < lim; ++b) {
                acc |= u64(vals[base + b]) << (b * bit_width);
            }
            // bit_width <= 8 => 8 values fit in <= 8 bytes = bit_width bytes!
            out.put(&acc, bit_width);
        }
        i += groups * 8;
    }
}

// Columnar buffer for per-thread user_daily output (grouped by person!)
struct UserDailyCols {
    // user_id dictionary: each person appears ONCE in dict_bytes, and run_lens[i] is
    // how many active days person i emitted!
    std::vector<u8> user_dict_bytes; // length-prefixed UTF-8 strings
    std::vector<u32> user_run_lens;  // number of rows for person i (1..ndays)

    std::vector<i32> day;
    std::vector<i32> sessions;
    std::vector<i32> events;
    std::vector<i64> revenue_cents;
    i64 max_rev = 0;

    // top_category: def_levels (0=null, 1=present) + non-null category IDs
    std::vector<u8> cat_def;
    std::vector<u16> cat_idx;

    void reserve(size_t est_rows, size_t est_people) {
        user_dict_bytes.reserve(est_people * 16);
        user_run_lens.reserve(est_people);
        day.reserve(est_rows);
        sessions.reserve(est_rows);
        events.reserve(est_rows);
        revenue_cents.reserve(est_rows);
        cat_def.reserve(est_rows);
        cat_idx.reserve(est_rows / 4 + 64);
    }

    inline void add_person_name(const u8* p, u32 len, u32 active_days) {
        size_t o = user_dict_bytes.size();
        user_dict_bytes.resize(o + 4 + len);
        std::memcpy(user_dict_bytes.data() + o, &len, 4);
        std::memcpy(user_dict_bytes.data() + o + 4, p, len);
        user_run_lens.push_back(active_days);
    }

    inline void add_row(i32 d, i32 s, i32 e, i64 rev, i32 top_cat) {
        day.push_back(d);
        sessions.push_back(s);
        events.push_back(e);
        revenue_cents.push_back(rev);
        if (rev > max_rev) max_rev = rev;
        if (top_cat >= 0) {
            cat_def.push_back(1);
            cat_idx.push_back(u16(top_cat));
        } else {
            cat_def.push_back(0);
        }
    }

    void write_parquet(const std::string& path, const std::vector<std::string>& categories) const {
        FileWriter fw(path);
        fw.write_bytes("PAR1", 4);

        const i64 nrows = i64(day.size());
        const bool rev_i32 = (max_rev <= INT32_MAX);
        std::vector<ColSchema> schema = {
            {"user_id",       TYPE_BYTE_ARRAY, CONV_UTF8, REP_REQUIRED},
            {"day",           TYPE_INT32,      CONV_DATE, REP_REQUIRED},
            {"sessions",      TYPE_INT32,      CONV_NONE, REP_REQUIRED},
            {"events",        TYPE_INT32,      CONV_NONE, REP_REQUIRED},
            {"revenue_cents", rev_i32 ? TYPE_INT32 : TYPE_INT64, CONV_NONE, REP_REQUIRED},
            {"top_category",  TYPE_BYTE_ARRAY, CONV_UTF8, REP_OPTIONAL},
        };

        std::vector<ColChunkMeta> metas;
        if (nrows > 0) {
            metas.reserve(6);
            Buf hdr, body;

            // 1. user_id: RLE_DICTIONARY (1 dict entry + 1 RLE run per person!)
            {
                ColChunkMeta cm;
                cm.phys_type = TYPE_BYTE_ARRAY;
                cm.name = "user_id";
                cm.encodings = {ENC_PLAIN, ENC_PLAIN_DICTIONARY, ENC_RLE};
                cm.num_values = nrows;
                cm.dict_page_offset = fw.pos();

                const u32 ndict = u32(user_run_lens.size());
                write_page_header_dict(hdr, i32(user_dict_bytes.size()), i32(ndict));
                fw.write_bytes(hdr.data.data(), hdr.size());
                fw.write_bytes(user_dict_bytes.data(), user_dict_bytes.size());

                cm.data_page_offset = fw.pos();
                u32 bw = (ndict <= 1) ? 0u : (32u - u32(__builtin_clz(ndict - 1u)));
                u32 val_bytes = (bw + 7u) >> 3;
                body.clear();
                body.reserve(1 + ndict * (val_bytes + 2));
                body.u8_put(u8(bw));
                if (bw == 0) {
                    body.uvarint(u64(nrows) << 1);
                } else {
                    for (u32 i = 0; i < ndict; ++i) {
                        body.uvarint(u64(user_run_lens[i]) << 1);
                        body.put(&i, val_bytes);
                    }
                }
                write_page_header_data(hdr, i32(body.size()), i32(nrows), ENC_PLAIN_DICTIONARY);
                fw.write_bytes(hdr.data.data(), hdr.size());
                fw.write_bytes(body.data.data(), body.size());
                cm.total_size = fw.pos() - cm.dict_page_offset;
                metas.push_back(std::move(cm));
            }

            // 2. day (INT32 DATE, PLAIN)
            auto write_i32_col = [&](const char* name, const std::vector<i32>& v) {
                ColChunkMeta cm;
                cm.phys_type = TYPE_INT32;
                cm.name = name;
                cm.encodings = {ENC_PLAIN, ENC_RLE};
                cm.num_values = nrows;
                cm.data_page_offset = fw.pos();
                i32 sz = i32(v.size() * 4);
                write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
                fw.write_bytes(hdr.data.data(), hdr.size());
                fw.write_bytes(v.data(), size_t(sz));
                cm.total_size = fw.pos() - cm.data_page_offset;
                metas.push_back(std::move(cm));
            };
            write_i32_col("day", day);
            write_i32_col("sessions", sessions);
            write_i32_col("events", events);

            // 5. revenue_cents (INT32 if fits else INT64, PLAIN)
            {
                ColChunkMeta cm;
                cm.name = "revenue_cents";
                cm.encodings = {ENC_PLAIN, ENC_RLE};
                cm.num_values = nrows;
                cm.data_page_offset = fw.pos();
                if (rev_i32) {
                    cm.phys_type = TYPE_INT32;
                    std::vector<i32> tmp(revenue_cents.size());
                    for (size_t i = 0; i < revenue_cents.size(); ++i) tmp[i] = i32(revenue_cents[i]);
                    i32 sz = i32(tmp.size() * 4);
                    write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
                    fw.write_bytes(hdr.data.data(), hdr.size());
                    fw.write_bytes(tmp.data(), size_t(sz));
                } else {
                    cm.phys_type = TYPE_INT64;
                    i32 sz = i32(revenue_cents.size() * 8);
                    write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
                    fw.write_bytes(hdr.data.data(), hdr.size());
                    fw.write_bytes(revenue_cents.data(), size_t(sz));
                }
                cm.total_size = fw.pos() - cm.data_page_offset;
                metas.push_back(std::move(cm));
            }

            // 6. top_category (OPTIONAL BYTE_ARRAY UTF8, RLE_DICTIONARY)
            {
                ColChunkMeta cm;
                cm.phys_type = TYPE_BYTE_ARRAY;
                cm.name = "top_category";
                cm.encodings = {ENC_PLAIN, ENC_PLAIN_DICTIONARY, ENC_RLE};
                cm.num_values = nrows;
                cm.dict_page_offset = fw.pos();

                Buf dict_buf;
                for (const auto& cat : categories) {
                    u32 len = u32(cat.size());
                    dict_buf.u32_le(len);
                    dict_buf.put(cat.data(), len);
                }
                write_page_header_dict(hdr, i32(dict_buf.size()), i32(categories.size()));
                fw.write_bytes(hdr.data.data(), hdr.size());
                fw.write_bytes(dict_buf.data.data(), dict_buf.size());

                cm.data_page_offset = fw.pos();
                encode_def_levels_1bit(cat_def, body);
                u32 ncats = u32(categories.size());
                u32 bw = (ncats <= 1) ? 0u : (32u - u32(__builtin_clz(ncats - 1u)));
                if (bw <= 8) {
                    encode_dict_indices_small(cat_idx, bw, body);
                } else {
                    // > 256 categories (impossible in normal datasets, exact RLE fallback)
                    u32 vb = (bw + 7u) >> 3;
                    body.u8_put(u8(bw));
                    for (u16 v : cat_idx) {
                        body.uvarint(2u); // run of 1
                        body.put(&v, vb);
                    }
                }
                write_page_header_data(hdr, i32(body.size()), i32(nrows), ENC_PLAIN_DICTIONARY);
                fw.write_bytes(hdr.data.data(), hdr.size());
                fw.write_bytes(body.data.data(), body.size());
                cm.total_size = fw.pos() - cm.dict_page_offset;
                metas.push_back(std::move(cm));
            }
        }

        write_footer(fw, schema, nrows, metas);
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
    fw.write_bytes("PAR1", 4);

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
    if (nrows > 0) {
        metas.reserve(7);
        Buf hdr, body;

        auto write_str_col = [&](const char* name, const std::vector<std::string>& dict) {
            ColChunkMeta cm;
            cm.phys_type = TYPE_BYTE_ARRAY;
            cm.name = name;
            cm.encodings = {ENC_PLAIN, ENC_RLE};
            cm.num_values = nrows;
            cm.data_page_offset = fw.pos();
            body.clear();
            body.reserve(size_t(nrows) * 12);
            for (const auto& r : rows) {
                const std::string& s = dict[r.camp_idx];
                u32 len = u32(s.size());
                body.u32_le(len);
                body.put(s.data(), len);
            }
            write_page_header_data(hdr, i32(body.size()), i32(nrows), ENC_PLAIN);
            fw.write_bytes(hdr.data.data(), hdr.size());
            fw.write_bytes(body.data.data(), body.size());
            cm.total_size = fw.pos() - cm.data_page_offset;
            metas.push_back(std::move(cm));
        };

        auto write_i32_col = [&](const char* name, auto getter) {
            ColChunkMeta cm;
            cm.phys_type = TYPE_INT32;
            cm.name = name;
            cm.encodings = {ENC_PLAIN, ENC_RLE};
            cm.num_values = nrows;
            cm.data_page_offset = fw.pos();
            std::vector<i32> vals(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) vals[i] = getter(rows[i]);
            i32 sz = i32(vals.size() * 4);
            write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
            fw.write_bytes(hdr.data.data(), hdr.size());
            fw.write_bytes(vals.data(), size_t(sz));
            cm.total_size = fw.pos() - cm.data_page_offset;
            metas.push_back(std::move(cm));
        };

        write_str_col("campaign_id", camp_ids);
        write_str_col("channel", camp_channels);
        write_i32_col("day", [](const CampaignDailyRow& r) { return r.day; });
        write_i32_col("sessions", [](const CampaignDailyRow& r) { return r.sessions; });
        write_i32_col("purchasing_sessions", [](const CampaignDailyRow& r) { return r.purchasing_sessions; });

        // revenue_cents (INT64 PLAIN)
        {
            ColChunkMeta cm;
            cm.phys_type = TYPE_INT64;
            cm.name = "revenue_cents";
            cm.encodings = {ENC_PLAIN, ENC_RLE};
            cm.num_values = nrows;
            cm.data_page_offset = fw.pos();
            std::vector<i64> vals(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) vals[i] = rows[i].revenue_cents;
            i32 sz = i32(vals.size() * 8);
            write_page_header_data(hdr, sz, i32(nrows), ENC_PLAIN);
            fw.write_bytes(hdr.data.data(), hdr.size());
            fw.write_bytes(vals.data(), size_t(sz));
            cm.total_size = fw.pos() - cm.data_page_offset;
            metas.push_back(std::move(cm));
        }

        write_i32_col("assisted_sessions", [](const CampaignDailyRow& r) { return r.assisted_sessions; });
    }

    write_footer(fw, schema, nrows, metas);
    fw.finish();
}

} // namespace pq
} // namespace hyp
