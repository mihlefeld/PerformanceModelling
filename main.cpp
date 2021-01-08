#include <iostream>
#include <cuda_runtime.h>
#include "tests.h"
#include "perf.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <filename> <gpuid>" << std::endl;
        return 0;
    }
    test();

    cudaSetDevice(atoi(argv[2]));

    auto measurements = load_from_file(argv[1]);
    find_hypothesis(measurements);
    matrix_free_cpu(measurements);
}
