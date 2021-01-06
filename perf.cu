#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#include "perf.h"
#include "matrix.h"
#include "common.h"

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

__device__ float* get_matrix_element_ptr(GPUMatrix m, int x, int y) {
    return (float*)((char*)m.elements + y * m.pitch) + x;
}

__device__ float get_matrix_element(GPUMatrix m, int x, int y) {
    float* pElement = (float*)((char*)m.elements + y * m.pitch) + x;
    return *pElement;
}

__device__ void set_matrix_element(GPUMatrix m, int x, int y, float v) {
    float* pElement = (float*)((char*)m.elements + y * m.pitch) + x;
    *pElement = v;
}

template<int D>
__device__ float evaluate_single(unsigned char *combination, float coef, float *ctps, float *params) {
    float prod = coef;
    for (int i = 0; i < D; i++) {
        prod *= combination[i] * pow(params[i], ctps[i*2]) * pow(log2(params[i]), ctps[i*2 + 1]);
    }
    return prod;
}

// coefs has to be initialized with all 1s
// coefs needs to be D + 1 in size
template<int D>
__device__ float evaluate_multi(unsigned char *combination, float *coefs, float *ctps, float *params) {
    float result = coefs[0];
    for (int i = 0; i < D; i++) {
        result += evaluate_single<D>(&combination[i*D], coefs[i + 1], ctps, params);;
    }
    return result;
}

template<int D>
__global__ void prepare_gels_batched(GPUMatrix measurements, int num_combinations, int num_buildingblocks, int num_hypothesis,
                                     float *amatrices, float *cmatrices, float **amptrs, float **cmptrs) {

    if constexpr(D == 2) {
        num_combinations = 4;
    } else if constexpr(D == 3) {
        num_combinations = 23;
    } else {
        num_combinations = 2;
    }

    // measurements should probably be a matrix
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_hypothesis)
        return;

    if(idx == 0) {
        printf("HELLO THERE\n");
    }

    if(idx == 1200000) {
        printf("HELLO THERE ME WERE NICE\n");
    }

    // TODO exit if index out of range
    float *amatrix = &amatrices[idx * (measurements.height * (D + 1))];
    float *cmatrix = &cmatrices[idx * measurements.height];
    amptrs[idx] = amatrix;
    cmptrs[idx] = cmatrix;
    float ctps[2*D];
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

    for (int i = 0; i < measurements.height; i++) {
        // first element in every row should be 1, since there's always a constant component
        amatrix[i] = 1;
        for (int j = 0; j < D; j++) {
            // this value needs to be written into a giant list of matrices
            float y = evaluate_single<D>(&combination[D*j], 1, ctps, get_matrix_element_ptr(measurements, 0, i));
            // danger danger, amatrix must be column major format
            amatrix[j * measurements.height + i + 1] = y;
        }
        cmatrix[i] = get_matrix_element(measurements, D, i);
    }
}

void find_hypothesis(const CPUMatrix &measurements) {
    GPUMatrix device_measurements = matrix_alloc_gpu(measurements.width, measurements.height);
    matrix_upload(measurements, device_measurements);


    int num_combinations = 333;
    int num_buildingblocks = 39;
    int num_hypothesis = -1;
    float *amatrices, *cmatrices, **amptrs, **cmptrs;


    int dimensions = measurements.width-1;
    switch(dimensions) {
        case 2:

            break;
        case 3:
            num_combinations = 23;
            // TODO num_buildingblocks anpassen
            num_hypothesis = pow(num_buildingblocks, dimensions) * num_combinations;

            cudaMalloc(&amatrices, num_hypothesis * measurements.height * (dimensions+1) * sizeof(float));
            CUDA_CHECK_ERROR;
            cudaMalloc(&cmatrices, num_hypothesis * measurements.height * sizeof(float));
            CUDA_CHECK_ERROR;
            cudaMalloc(&amptrs, num_hypothesis * sizeof(float*));
            CUDA_CHECK_ERROR;
            cudaMalloc(&cmptrs, num_hypothesis * sizeof(float*));
            CUDA_CHECK_ERROR;

            prepare_gels_batched<3><<<div_up(num_hypothesis, 1024), 1024>>>(device_measurements, num_combinations, num_buildingblocks, num_hypothesis, amatrices, cmatrices, amptrs, cmptrs);
            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR;

            break;
        case 4:

            break;
        case 5:

            break;

        default:
            std::cerr << "Finding hypothesis with dimensions " << dimensions << " is not supported!" << std::endl;
            exit(EXIT_FAILURE);
    }

    cudaFree(amatrices);
    CUDA_CHECK_ERROR;
    cudaFree(cmatrices);
    CUDA_CHECK_ERROR;
    cudaFree(amptrs);
    CUDA_CHECK_ERROR;
    cudaFree(cmptrs);
    CUDA_CHECK_ERROR;
    matrix_free_gpu(device_measurements);
    // TODO return the hypothesis
}
