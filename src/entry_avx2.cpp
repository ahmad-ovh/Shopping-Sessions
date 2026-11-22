#define HYP_NS hyp_avx2
#include "pipeline.hpp"

void run_pipeline_avx2(int argc, char** argv) {
    static hyp_avx2::Pipeline pipe;
    pipe.run(argc, argv);
}
