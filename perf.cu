#include "perf.h"
#include "matrix.h"
#include <cstdio>

__constant__ float building_blocks[] = {
        0, 1,
        0, 2,
        1./4, 0,
        1./3, 0,
        1./4, 1,
        1./3, 1,
        1./4, 2,
        1./3, 2,
        1./2, 0,
        1./2, 1,
        1./2, 2,
        2./3, 0,
        3./4, 0,
        2./3, 1,
        3./4, 1,
        4./5, 0,
        2./3, 2,
        3./4, 2,
        1, 0,
        1, 1,
        1, 2,
        5./4, 0,
        5./4, 1,
        4./3, 0,
        4./3, 1,
        3./2, 0,
        3./2, 1,
        3./2, 2,
        5./3, 0,
        7./4, 0,
        2, 0,
        2, 1,
        2, 2,
        9./4, 0,
        7./4, 0,
        2, 0,
        2, 1,
        2, 2,
        9./4, 0,
        7./3, 0,
        5./2, 0,
        5./2, 1,
        5./2, 2,
        8./3, 0,
        11./4, 0,
        3, 0,
        3, 1
};

__constant__ unsigned char combinations[23*3*3];

unsigned char combinations_2d[] {
        1, 1, 0, 0,
        0, 1, 1, 0,
        1, 1, 1, 0,
        1, 1, 0, 1
};

unsigned char combinations_3d[] {
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

unsigned char combinations_4d[] {
    1, 1, 1, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,

    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
};

unsigned char combinations_5d[] {
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,


    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1
};

template<int D>
__device__ float evaluate_single(unsigned char *combination, float coef, float *ctps, float *params) {
    float prod = coef;
    for (int i = 0; i < D; i++) {
        prod *= combination[i] * pow(params[i], ctps[i*2]) * pow(log2(params[i]), ctps[i*2 + 1]);
    }
    return prod;
}

template<int D>
__device__ float evaluate_multi(unsigned char *combination, float *coefs, float *ctps, float *params) {
    float result = 0;
    for (int i = 0; i < D; i++) {
        result += evaluate_single<D>(&combination[i*D], coefs[i], ctps, params);;
    }
    return result;
}

template<int D>
__global__ void find_best_hypothesis(GPUMatrix measurements, int num_combinations, int num_buildingblocks) {
    // measurements should probably be a matrix
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float ctps[2*D];
    float coefs[D];
    unsigned char combination[D*D];
    /*
     * a*b*c + a*b + c
     * 1 1 1
     * 1 1 0
     * 0 0 1
     */

    int combination_index = idx / num_combinations;
    idx = combination_index % num_combinations;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            combination[i*D + j] = combinations[combination_index * D * D + i * D + j];
        }
    }

    int r = pow(num_buildingblocks, D-1);
    for (int i = D - 1; i >= 0; i--) {
        int ctpi = idx / r;
        idx = ctpi % r;
        for (int j = 0; j < 2; j++) {
            ctps[i * 2 + j] = building_blocks[ctpi * 2 + j];
        }
        r/= num_buildingblocks;
    }

    // compute coefficients


    // cross validation


    // parallel reduction


    // write block result
}