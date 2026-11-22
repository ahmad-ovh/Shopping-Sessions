#include <cstdlib>
#include <cstring>

void run_pipeline_avx2(int argc, char** argv);
void run_pipeline_avx512(int argc, char** argv);

int main(int argc, char** argv) {
    const char* force = std::getenv("HYPERION_ISA");
    bool use_avx512 = false;
#if defined(__x86_64__)
    if (!force || std::strcmp(force, "avx2") != 0) {
        use_avx512 = __builtin_cpu_supports("avx512f") &&
                     __builtin_cpu_supports("avx512bw") &&
                     __builtin_cpu_supports("avx512vl");
    }
#endif
    if (use_avx512) {
        run_pipeline_avx512(argc, argv);
    } else {
        run_pipeline_avx2(argc, argv);
    }
    return 0;
}
