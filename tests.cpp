#include <cmath>
#include <iostream>

unsigned char combinations_t3d[] {
        // x*y*z
        1, 1, 1,
        0, 0, 0,
        0, 0, 0,

        // x+y+z
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        // x*y*z + x
        1, 1, 1,
        1, 0, 0,
        0, 0, 0,

        // x*y*z + y
        1, 1, 1,
        0, 1, 0,
        0, 0, 0,

        // x*y*z + z
        1, 1, 1,
        0, 0, 1,
        0, 0, 0,

        // x*y*z + x*y
        1, 1, 1,
        1, 1, 0,
        0, 0, 0,

        // x*y*z + y*z
        1, 1, 1,
        0, 1, 1,
        0, 0, 0,

        // x*y*z + x*z
        1, 1, 1,
        1, 0, 1,
        0, 0, 0,

        // x*y*z + x*y + z
        1, 1, 1,
        1, 1, 0,
        0, 0, 1,

        // x*y*z + y*z + x
        1, 1, 1,
        0, 1, 1,
        1, 0, 0,

        // x*y*z + x*z + y
        1, 1, 1,
        1, 0, 1,
        0, 1, 0,

        // x*y*z + x + y
        1, 1, 1,
        1, 0, 0,
        0, 1, 0,

        // x*y*z + x + z
        1, 1, 1,
        1, 0, 0,
        0, 0, 1,

        // x*y*z + y + z
        1, 1, 1,
        0, 1, 0,
        0, 0, 1,

        // x*y + z
        1, 1, 0,
        0, 0, 1,
        0, 0, 0,

        // x*y + z + y
        1, 1, 0,
        0, 0, 1,
        0, 1, 0,

        // x*y + z + x
        1, 1, 0,
        0, 0, 1,
        1, 0, 0,

        // x*z + y
        1, 0, 1,
        0, 1, 0,
        0, 0, 0,

        // x*z + y + x
        1, 0, 1,
        0, 1, 0,
        1, 0, 0,

        // x*z + x
        0, 1, 1,
        1, 0, 0,
        0, 0, 0,

        // y*z + x
        0, 1, 1,
        1, 0, 0,
        0, 0, 0,

        // y*z + x + y
        0, 1, 1,
        1, 0, 0,
        0, 1, 0,

        // y*z + x + z
        0, 1, 1,
        1, 0, 0,
        0, 0, 1,
};

template<int D>
float evaluate_single(unsigned char *combination, float coef, float *ctps, float *params) {
    float prod = coef;
    // if the combination is 0 0 0, zero should be returned, instead of prod
    bool nonzero = 0;
    for (int i = 0; i < D; i++) {
        nonzero |= combination[i];
        if (combination[i])
            prod *= std::pow(params[i], ctps[i*2]) * std::pow(std::log2(params[i]), ctps[i*2 + 1]);
    }
    return nonzero ? prod : 0;
}

void test() {
    // test (p^0 * log2(p)^1) * log2(q)^1 * log2(r)^1
    // pqr = (60, 5, 500)
    // 122.96909567179195
    float ctps[] = {
            0, 1, 0, 1, 0, 1
    };
    float params[] = {
            60, 5, 500
    };
    float expected = 122.96909567179195;
    float result = evaluate_single<3>(&combinations_t3d[0], 1, ctps, params);
    std::cout << "Expected: " << expected << std::endl;
    std::cout << "Got: " << result << std::endl;

}