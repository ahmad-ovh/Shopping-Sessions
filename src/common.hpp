#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#ifndef HYP_NS
#define HYP_NS hyp
#endif

namespace HYP_NS {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

inline constexpr i64 SEC_MS = 1000LL;
inline constexpr i64 MIN_MS = 60LL * SEC_MS;
inline constexpr i64 HOUR_MS = 60LL * MIN_MS;
inline constexpr i64 DAY_MS = 24LL * HOUR_MS;
inline constexpr i64 WEEK_MS = 7LL * DAY_MS;
inline constexpr i64 SESSION_GAP_MS = 30LL * MIN_MS;
inline constexpr u32 DEAD_TS = 0xFFFFFFFFu;
inline constexpr u32 INVALID_LOC = 0xFFFFFFFFu;

[[noreturn]] inline void die(const char* msg) {
    std::fprintf(stderr, "hyperion fatal: %s (errno=%d: %s)\n", msg, errno, std::strerror(errno));
    _exit(2);
}

inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

struct Profiler {
    bool enabled = false;
    double t0 = 0.0;
    double last = 0.0;
    void init() {
        const char* e = std::getenv("HYPERION_PROFILE");
        enabled = (e && e[0] && e[0] != '0');
        t0 = last = now_sec();
    }
    void mark(const char* label) {
        if (!enabled) return;
        double cur = now_sec();
        std::fprintf(stderr, "[prof] %-24s +%7.2f ms (total %7.2f ms)\n",
                     label, (cur - last) * 1000.0, (cur - t0) * 1000.0);
        last = cur;
    }
    void note(const char* label, double ms) const {
        if (!enabled) return;
        std::fprintf(stderr, "  [sub] %-23s %7.2f ms\n", label, ms);
    }
};

// Fast 128-bit wyhash-style mixer (4 cycles, full 64-bit avalanche)
static inline u64 wyhash64(u64 x) {
    __uint128_t r = __uint128_t(x ^ 0xa0761d6478bd642fULL) *
                    __uint128_t(x ^ 0xe7037ed1a0b428dbULL);
    return u64(r) ^ u64(r >> 64);
}

static inline u64 wyhash128(u64 a, u64 b) {
    __uint128_t r = __uint128_t(a ^ 0xa0761d6478bd642fULL) *
                    __uint128_t(b ^ 0xe7037ed1a0b428dbULL);
    return u64(r) ^ u64(r >> 64);
}

static inline u64 wyhash_bytes(const u8* p, size_t len) {
    u64 h = 0x2d358dccaa6c78a5ULL ^ (u64(len) * 0xa0761d6478bd642fULL);
    size_t i = 0;
    while (i + 8 <= len) {
        u64 w;
        std::memcpy(&w, p + i, 8);
        h = wyhash128(h ^ w, w ^ 0x8bb84b93962eacc9ULL);
        i += 8;
    }
    if (i < len) {
        u64 tail = 0;
        std::memcpy(&tail, p + i, len - i);
        h = wyhash128(h ^ tail, (u64(len) << 32) ^ 0x4b33a62ed433d4a3ULL);
    }
    return wyhash64(h);
}

static inline u32 fastrange32(u32 hash, u32 range) {
    return u32((u64(hash) * u64(range)) >> 32);
}

// Runtime THP safety check: avoid MADV_HUGEPAGE if THP is disabled
inline bool thp_safe() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* env = std::getenv("HYPERION_NOHUGE");
    if (env && env[0] == '1') {
        cached = 0;
        return false;
    }
    int fd = ::open("/sys/kernel/mm/transparent_hugepage/enabled", O_RDONLY);
    if (fd < 0) {
        cached = 1;
        return true;
    }
    char buf[128] = {};
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n > 0 && std::strstr(buf, "[never]")) {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

// Track anonymous mmap regions so we can parallel-MADV_DONTNEED before _exit
struct AnonRegion {
    void* ptr;
    size_t len;
};

struct RegionTracker {
    std::atomic<u32> count{0};
    AnonRegion regs[4096];
    void record(void* p, size_t n) {
        u32 idx = count.fetch_add(1, std::memory_order_relaxed);
        if (idx < 4096) regs[idx] = {p, n};
    }
};

inline RegionTracker g_regions;

inline void* huge_alloc(size_t bytes, bool track = true, void** out_raw = nullptr, size_t* out_raw_len = nullptr) {
    if (bytes == 0) bytes = 4096;
    constexpr size_t HP = 2u << 20;
    size_t rounded = (bytes + 4095u) & ~size_t(4095u);
    void* p = nullptr;
    size_t alloc_len = rounded;
    if (rounded >= HP && thp_safe()) {
        alloc_len = rounded + HP;
        void* raw = ::mmap(nullptr, alloc_len, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED) die("mmap huge_alloc");
        uintptr_t addr = (uintptr_t(raw) + HP - 1) & ~(uintptr_t(HP) - 1);
        p = reinterpret_cast<void*>(addr);
#ifdef MADV_HUGEPAGE
        ::madvise(p, rounded, MADV_HUGEPAGE);
#endif
        if (out_raw) *out_raw = raw;
        if (out_raw_len) *out_raw_len = alloc_len;
        if (track) g_regions.record(raw, alloc_len);
        return p;
    }
    p = ::mmap(nullptr, alloc_len, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) die("mmap alloc");
    if (out_raw) *out_raw = p;
    if (out_raw_len) *out_raw_len = alloc_len;
    if (track) g_regions.record(p, alloc_len);
    return p;
}

// Growable raw buffer backed by huge_alloc (MAP_NORESERVE): zero glibc 4KB faults, zero resize zero-fill!
template <class T>
struct HugeBuf {
    T* ptr = nullptr;
    size_t len = 0;
    size_t cap = 0;

    void init_reserve(size_t reserve_elems) {
        if (ptr) return;
        cap = reserve_elems < 1024 ? 1024 : reserve_elems;
        ptr = static_cast<T*>(huge_alloc(cap * sizeof(T)));
        len = 0;
    }

    inline void grow_to(size_t need) {
        size_t ncap = cap ? cap : 1024;
        while (ncap < need) ncap <<= 1;
        T* np = static_cast<T*>(huge_alloc(ncap * sizeof(T)));
        if (len > 0 && ptr) std::memcpy(np, ptr, len * sizeof(T));
        ptr = np;
        cap = ncap;
    }

    inline void ensure_extra(size_t extra) {
        if (__builtin_expect(len + extra > cap, 0)) grow_to(len + extra);
    }

    inline void push(const T& v) {
        if (__builtin_expect(len == cap, 0)) grow_to(len + 1);
        ptr[len++] = v;
    }

    inline T* data() { return ptr; }
    inline const T* data() const { return ptr; }
    inline size_t size() const { return len; }
    inline bool empty() const { return len == 0; }
    inline void clear() { len = 0; }
};

// Bump-pointer block allocator backed by MAP_NORESERVE huge-page virtual regions
// Supports rewind() to recycle already-faulted 2 MiB THP pages across disjoint pipeline phases!
struct SlabArena {
    u8* cur = nullptr;
    u8* end = nullptr;
    u8* first_cur = nullptr;
    u8* first_end = nullptr;
    size_t reserve_bytes = 48ULL << 20;

    void set_chunk_size(size_t bytes) {
        reserve_bytes = (bytes + (2u << 20) - 1) & ~size_t((2u << 20) - 1);
        if (reserve_bytes < (2u << 20)) reserve_bytes = 2u << 20;
    }

    void rewind() {
        if (first_cur) {
            cur = first_cur;
            end = first_end;
        }
    }

    void* alloc(size_t bytes, size_t align = 64) {
        uintptr_t p = (uintptr_t(cur) + align - 1) & ~(uintptr_t(align) - 1);
        if (__builtin_expect(p + bytes > uintptr_t(end), 0)) {
            size_t need = bytes + align;
            size_t sz = need > reserve_bytes ? ((need + (2u << 20) - 1) & ~size_t((2u << 20) - 1)) : reserve_bytes;
            cur = static_cast<u8*>(huge_alloc(sz));
            end = cur + sz;
            if (!first_cur) {
                first_cur = cur;
                first_end = end;
            }
            p = (uintptr_t(cur) + align - 1) & ~(uintptr_t(align) - 1);
        }
        cur = reinterpret_cast<u8*>(p + bytes);
        return reinterpret_cast<void*>(p);
    }
};

// Memory-mapped input file with >= 64 bytes of zeroed guard padding past EOF
struct MFile {
    const u8* data = nullptr;
    size_t size = 0;
    size_t mapped = 0;
    int fd = -1;

    bool open_map(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (::fstat(fd, &st) != 0) { ::close(fd); fd = -1; return false; }
        size = size_t(st.st_size);
        if (size == 0) {
            static const u8 empty_pad[128] = {};
            data = empty_pad;
            mapped = 0;
            ::close(fd);
            fd = -1;
            return true;
        }
        size_t file_pages = (size + 4095u) & ~size_t(4095u);
        mapped = file_pages + 4096u;
        void* base = ::mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) die("mmap reserve");
        void* m = ::mmap(base, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
        if (m == MAP_FAILED) die("mmap file");
        data = static_cast<const u8*>(base);
        return true;
    }

    void close_fd() {
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
};

// Persistent low-latency thread pool using spin + futex wake
class ThreadPool {
    int nt_ = 1;
    std::vector<std::thread> workers_;
    std::function<void(int)> fn_;
    alignas(64) std::atomic<u32> gen_{0};
    alignas(64) std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};

    static void futex_wait(std::atomic<u32>& a, u32 expected) {
        syscall(SYS_futex, reinterpret_cast<u32*>(&a), FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
    }
    static void futex_wake_all(std::atomic<u32>& a) {
        syscall(SYS_futex, reinterpret_cast<u32*>(&a), FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0);
    }

public:
    explicit ThreadPool(int nthreads) : nt_(nthreads < 1 ? 1 : (nthreads > 16 ? 16 : nthreads)) {
        workers_.reserve(nt_ - 1);
        for (int t = 1; t < nt_; ++t) {
            workers_.emplace_back([this, t]() {
                u32 seen = 0;
                for (;;) {
                    u32 g = gen_.load(std::memory_order_acquire);
                    if (g == seen) {
                        for (int spin = 0; spin < 2000; ++spin) {
                            __builtin_ia32_pause();
                            g = gen_.load(std::memory_order_acquire);
                            if (g != seen) break;
                        }
                        while (g == seen) {
                            futex_wait(gen_, seen);
                            g = gen_.load(std::memory_order_acquire);
                        }
                    }
                    seen = g;
                    if (stop_.load(std::memory_order_acquire)) break;
                    fn_(t);
                    done_.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }

    int size() const { return nt_; }

    template <class F>
    void run(F&& f) {
        if (nt_ == 1) {
            f(0);
            return;
        }
        fn_ = std::forward<F>(f);
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        futex_wake_all(gen_);
        fn_(0);
        while (done_.load(std::memory_order_acquire) < nt_ - 1) {
            __builtin_ia32_pause();
        }
    }

    void detach_all() {
        for (auto& w : workers_) {
            if (w.joinable()) w.detach();
        }
    }
};

} // namespace HYP_NS
