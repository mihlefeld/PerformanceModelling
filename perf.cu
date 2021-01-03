//
// Created by thesh on 14/12/2020.
//
#include "perf.h"
#include <cstdio>

__constant__ float building_blocks[2*43];
__constant__ unsigned char combinations[23*3*3];

template<int D>
__global__ void hello_world_kernel() {
    printf("%d: %d", D, threadIdx.x);
}

template<int D>
__device__ float evaluate(unsigned char *combination, float *coefs, float *ctps, float *params) {
    float result = 0;
    for (int i = 0; i < D; i++) {
        float prod = coefs[i];
        float param = params[i];
        for (int j = 0; j < D; j++) {
            prod *= pow(param, ctps[j*2]) + pow(log2(param), ctps[j*2 + 1]);
        }
        result += prod;
    }
    return result;
}

template<int D>
__global__ void find_best_hypothesis(float *measurements, int num_combinations, int num_buildingblocks) {
    // measurements should probably be a matrix
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float ctps[2*D];
    float coefs[D];
    unsigned char combination[D][D];

    int combination_index = idx / num_combinations;
    idx = combination_index % num_combinations;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            combination[i][j] = combinations[combination_index * D * D + i * D + j];
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

void hello_world() {
    hello_world_kernel<3><<<1, 1>>>();
}