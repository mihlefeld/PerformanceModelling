#define DEBUG
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#include "cublas_v2.h"
#include "perf.h"
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
        if (combination[i])
            prod *= pow(params[i], ctps[i*2]) * pow(log2(params[i]), ctps[i*2 + 1]);
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
    // measurements should probably be a matrix
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_hypothesis)
        return;

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
    int mod_idx = combination_index % num_combinations;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            combination[i*D + j] = combinations[combination_index * D * D + i * D + j];
        }
    }

    int r = pow(num_buildingblocks, D-1);
    for (int i = D - 1; i >= 0; i--) {
        int ctpi = mod_idx / r;
        mod_idx = ctpi % r;
        for (int j = 0; j < 2; j++) {
            ctps[i * 2 + j] = building_blocks[ctpi * 2 + j];
        }
        r/= num_buildingblocks;
    }

#ifdef DEBUG
    if (idx == 0) {
        float *ptr = get_matrix_element_ptr(measurements, 0, 0);
        for (int i = 0; i < D; i++) {
            printf("%f, ", ptr[i]);
        }
        printf("\nctps: ");
        for (int i = 0; i < D; i++) {
            printf("(%f, %f), ", ctps[i*2], ctps[i*2 + 1]);
        }
        printf("\nevalr: ");
        for (int j = 0; j < D; j++) {
            // this value needs to be written into a giant list of matrices
            float y = evaluate_single<D>(&combination[D * j], 1, ctps, get_matrix_element_ptr(measurements, 0, 0));
            printf("%f, ", y);
        }
        printf("\ncomb:\n");
        for (int i = 0; i < D; i++) {
            unsigned char *comptr = &combination[D * i];
            for (int j = 0; j < D; j++) {
                printf("%d, ", comptr[j]);
            }
            printf("\n");
        }
    }
#endif

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
    cudaSetDevice(5);

    cublasHandle_t handle;

    GPUMatrix device_measurements = matrix_alloc_gpu(measurements.width, measurements.height);
    matrix_upload(measurements, device_measurements);


    CPUMatrix A, X, C;
    int num_combinations = 333;
    int num_buildingblocks = 39;
    int num_hypothesis = -1;
    float *amatrices, *cmatrices, **amptrs, **cmptrs;
    int dimensions = measurements.width-1;
    auto *yvec = new float[measurements.height];
    auto *amat = new float[(dimensions + 1) * measurements.height];
    auto *coefs = new float[dimensions + 1];
    int info;
    switch(dimensions) {
        case 2:

            break;
        case 3:
            num_combinations = 23;
            // TODO num_buildingblocks anpassen
            num_hypothesis = pow(num_buildingblocks, dimensions) * num_combinations;

            cudaMemcpyToSymbol(combinations, combinations_3d, num_combinations, cudaMemcpyHostToDevice);

            cudaMalloc(&amatrices, num_hypothesis * measurements.height * (dimensions+1) * sizeof(float));
            CUDA_CHECK_ERROR;
            cudaMalloc(&cmatrices, num_hypothesis * measurements.height * sizeof(float));
            CUDA_CHECK_ERROR;
            cudaMalloc(&amptrs, num_hypothesis * sizeof(float*));
            CUDA_CHECK_ERROR;
            cudaMalloc(&cmptrs, num_hypothesis * sizeof(float*));
            CUDA_CHECK_ERROR;

            prepare_gels_batched<3><<<div_up(num_hypothesis, 512), 512>>>(device_measurements, num_combinations, num_buildingblocks, num_hypothesis, amatrices, cmatrices, amptrs, cmptrs);
            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR;

            cudaMemcpy(yvec, cmatrices, measurements.height * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(amat, amatrices, measurements.height * (dimensions + 1) * sizeof(float), cudaMemcpyDeviceToHost);

            CUBLAS_CALL(cublasCreate(&handle));
            CUBLAS_CALL(cublasSgelsBatched(
                    handle,
                    CUBLAS_OP_N,
                    measurements.height, // height of Aarray
                    dimensions + 1, // width of Aarray and height of Carray
                    1, // width of Carray
                    amptrs, // Aarray pointer
                    measurements.height, // lda >= max(1,m)
                    cmptrs, // Carray pointer
                    measurements.height, // ldc >= max(1,m)
                    &info,
                    nullptr,
                    num_hypothesis
                )
            )

            cudaMemcpy(coefs, cmatrices, (dimensions + 1) * sizeof(float), cudaMemcpyDeviceToHost);

            X.width = 1; X.height = dimensions + 1; X.elements = coefs;
            C.width = 1; C.height = measurements.height; C.elements = yvec;
            A.width = measurements.height; A.height = dimensions + 1; A.elements = amat;
            printf("A:\n");
            matrix_print(A);
            printf("X:\n");
            matrix_print(X);
            printf("C:\n");
            matrix_print(C);
            printf("Done");



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
