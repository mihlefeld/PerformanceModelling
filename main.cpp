#include <iostream>
#include "perf.h"
#include "matrix.h"

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 0;
    }

    auto touple = load_from_file(argv[1]);

    for(int i = 0; i < touple.first.height; i++) {
        for(int j = 0; j < touple.first.width; j++) {
            std::cout << touple.first.elements[i*touple.first.width + j] << " ";
        }
        std::cout << touple.second.elements[i] << std::endl;
    }

    //hello_world();
    return 0;
}
