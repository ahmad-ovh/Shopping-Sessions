#define HYP_NS hyp_avx512
#include "pipeline.hpp"

void run_pipeline_avx512(int argc, char** argv) {
    static hyp_avx512::Pipeline pipe;
    pipe.run(argc, argv);
}
