#include <iostream>
#include "perf.h"
#include "matrix.h"

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 0;
    }

    load_from_file(argv[1]);

    //hello_world();
    return 0;
}
