#include <iostream>
#include "tests.h"
#include "perf.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 0;
    }
    test();

    auto measurements = load_from_file(argv[1]);
    find_hypothesis(measurements);
    matrix_free_cpu(measurements);
}
