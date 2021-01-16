#include <iostream>
#include <cuda_runtime.h>
#include "../test/tests.h"
#include "perf.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <filename> [gpuid]" << std::endl;
        return 0;
    }

    int gpu = 0;
    if (argc == 3) {
        gpu = atoi(argv[2]);
    }

    auto measurements = load_from_file(argv[1]);
    auto hypothesis = find_hypothesis(measurements, 0.5, gpu);
    hypothesis.print();
    matrix_free_cpu(measurements);
}
