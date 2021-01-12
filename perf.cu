#define DEBUG
#define DEBUG_IDX 0 * 59319
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#include "cublas_v2.h"
#include "perf.h"
#include "common.h"

__constant__ float building_blocks[] = {
        0, 0,
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
        7./3, 0,
        5./2, 0,
        5./2, 1,
        5./2, 2,
        8./3, 0,
        11./4, 0,
        3, 0,
        3, 1
};

__constant__ unsigned char combinations[256];

int combinations_2d_end_indices[] {
    1, 4
};

unsigned char combinations_2d[] {
        1, 1,
        0, 0,

        0, 1,
        1, 0,

        1, 1,
        1, 0,

        1, 1,
        0, 1
};

// combination counts: element at index i determines where the combinations
// with i + 2 columns start if -1, the combinations using that many columns are not present

int combinations_3d_end_indices[] {
    1, 11, 23
};

unsigned char combinations_3d[] {
    // 1c: x*y*z
    1, 1, 1,
    0, 0, 0,
    0, 0, 0,

    // 2c: x*y*z + x
    1, 1, 1,
    1, 0, 0,
    0, 0, 0,

    // 2c: x*y*z + y
    1, 1, 1,
    0, 1, 0,
    0, 0, 0,

    // 2c: x*y*z + z
    1, 1, 1,
    0, 0, 1,
    0, 0, 0,

    // 2c: x*y*z + x*y
    1, 1, 1,
    1, 1, 0,
    0, 0, 0,

    // 2c: x*y*z + y*z
    1, 1, 1,
    0, 1, 1,
    0, 0, 0,

    // 2c: x*y*z + x*z
    1, 1, 1,
    1, 0, 1,
    0, 0, 0,

    // 2c: x*y + z
    1, 1, 0,
    0, 0, 1,
    0, 0, 0,

    // 2c: x*z + y
    1, 0, 1,
    0, 1, 0,
    0, 0, 0,

    // 2c: x*z + x
    0, 1, 1,
    1, 0, 0,
    0, 0, 0,

    // 2c: y*z + x
    0, 1, 1,
    1, 0, 0,
    0, 0, 0,

    // 3c: x+y+z
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,

    // 3c: x*y*z + x*y + z
    1, 1, 1,
    1, 1, 0,
    0, 0, 1,

    // 3c: x*y*z + y*z + x
    1, 1, 1,
    0, 1, 1,
    1, 0, 0,

    // 3c: x*y*z + x*z + y
    1, 1, 1,
    1, 0, 1,
    0, 1, 0,

    // 3c: x*y*z + x + y
    1, 1, 1,
    1, 0, 0,
    0, 1, 0,

    // 3c: x*y*z + x + z
    1, 1, 1,
    1, 0, 0,
    0, 0, 1,

    // 3c: x*y*z + y + z
    1, 1, 1,
    0, 1, 0,
    0, 0, 1,

    // 3c: x*y + z + y
    1, 1, 0,
    0, 0, 1,
    0, 1, 0,

    // 3c: x*y + z + x
    1, 1, 0,
    0, 0, 1,
    1, 0, 0,

    // 3c: x*z + y + x
    1, 0, 1,
    0, 1, 0,
    1, 0, 0,

    // 3c: y*z + x + y
    0, 1, 1,
    1, 0, 0,
    0, 1, 0,

    // 3c: y*z + x + z
    0, 1, 1,
    1, 0, 0,
    0, 0, 1,
};

int combinations_4d_end_indices[] {
    1, -1, -1, 2
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

int combinations_5d_end_indices[] {
    1, -1, -1, -1, 2
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
    // if the combination is 0 0 0, zero should be returned, instead of prod
    bool nonzero = 0;
    for (int i = 0; i < D; i++) {
        nonzero |= combination[i];
        if (combination[i])
            prod *= pow(params[i], ctps[i*2]) * pow(log2(params[i]), ctps[i*2 + 1]);
    }
    return nonzero ? prod : 0;
}

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
__device__ void get_data_from_indx(int idx, float *ctps, unsigned char **combination,
                                   int num_combinations, int num_buildingblocks, int num_hypothesis) {
    int div_ci = num_hypothesis / num_combinations;
    int combination_index = idx / div_ci;
    int mod_idx = idx  % div_ci;
    *combination = &combinations[combination_index * D * D];

    int r = pow(num_buildingblocks, D-1);
    for (int i = D - 1; i >= 0; i--) {
        int ctpi = mod_idx / r;
        mod_idx = mod_idx % r;
        for (int j = 0; j < 2; j++) {
            ctps[i * 2 + j] = building_blocks[ctpi * 2 + j];
        }
        r/= num_buildingblocks;
    }
}

template<int D>
__global__ void __launch_bounds__(256) prepare_gels_batched(GPUMatrix measurements, int num_combinations, int num_buildingblocks, int num_hypothesis,
                                     float *amatrices, float *cmatrices, float **amptrs, float **cmptrs, int swap_indx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_hypothesis)
        return;

    float *amatrix = &amatrices[idx * (measurements.height * (D + 1))];
    float *cmatrix = &cmatrices[idx * measurements.height];
    amptrs[idx] = amatrix;
    cmptrs[idx] = cmatrix;
    // TODO: this really shouldn't be here, I don't even know why it's faster tbh
    __shared__ float sctps[512][2*D];
    float *ctps = sctps[threadIdx.x];
    unsigned char *combination;

    get_data_from_indx<D>(idx, ctps, &combination, num_combinations, num_buildingblocks, num_hypothesis);

    for (int i = 0; i < measurements.height; i++) {
        // first element in every row should be 1, since there's always a constant component
        amatrix[i] = 1;
    }

    for (int i = 0; i < measurements.height; i++) {
        // TODO: seperate coordinates and values
        int ii = i == swap_indx ? (measurements.height - 1) : (i == measurements.height - 1 ? swap_indx : i);
        cmatrix[i] = get_matrix_element(measurements, D, ii);
    }

    // TODO: this needs to be adapted to write 128 values at a time to allow for larger measurements matrices
    float column_cache[125];
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < measurements.height; i++) {
            int ii = i == swap_indx ? (measurements.height - 1) : (i == measurements.height - 1 ? swap_indx : i);
            // this value needs to be written into a giant list of matrices
            column_cache[i] = evaluate_single<D>(&combination[D*j], 1, ctps, get_matrix_element_ptr(measurements, 0, ii));
        }
        for (int i = 0; i < measurements.height; i++) {
            // danger danger, amatrix must be column major format
            amatrix[(j + 1) * measurements.height + i] = column_cache[i];
        }
    }
}

__device__ float smape(float pred, float actual) {
    float abssum = abs(pred) + abs(actual);
    return abssum != 0 ? 200 * (abs(pred - actual) / abssum) : 0;
}

__device__ float rss(float pred, float actual) {
    return pow(pred - actual, 2);
}

template<int D>
__global__ void compute_costs(GPUMatrix measurements, int num_combinations, int num_buildingblocks, int num_hypothesis,
                              float *cmatrices, float *rss_costs, float *smape_costs,
                              int validation_index) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_hypothesis)
        return;
    float ctps[2*D];
    float *coefs = &cmatrices[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx, ctps, &combination, num_combinations, num_buildingblocks, num_hypothesis);

    // assume the validation measurement is the last row in the matrix
    int i = validation_index;
    float *row_ptr = get_matrix_element_ptr(measurements, 0, i);
    float actual = row_ptr[D];
    float predicted = evaluate_multi<D>(combination, coefs, ctps, row_ptr);

    // TODO: investigate why the computed smape seems to be much lower than the final SMAPE
    rss_costs[idx] += rss(predicted, actual);
    smape_costs[idx] += smape(predicted, actual) / (measurements.height - 1);
}

template<int D>
__global__ void print_hypothesis(int minimum_rss_cost_idx, int num_combinations, int num_buildingblocks, int num_hypothesis, GPUMatrix measurements, float* cmatrices, float *smape_costs) {
    int idx = minimum_rss_cost_idx;

    float ctps[2*D];
    float *coefs = &cmatrices[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx, ctps, &combination, num_combinations, num_buildingblocks, num_hypothesis);

    printf("\ncoefs: ");
    for(int i = 0; i < D+1; i++) {
        printf("%f, ", coefs[i]);
    }
    printf("\nctps: ");
    for(int i = 0; i < D; i++) {
        printf("(%f, %f), ", ctps[2*i], ctps[2*i+1]);
    }
    printf("\ncombination: \n");
    for(int i = 0; i < D; i++) {
        printf("(");
        for (int j = 0; j < D-1; j++) {
            printf("%d, ", combination[i*D + j]);
        }
        printf("%d)\n", combination[i*D + D - 1]);
    }
    printf("Cross Validation SMAPE: %f\n", smape_costs[idx]);
    printf("\n\n");
}

template<int D>
void find_hypothesis_templated(
        int num_buildingblocks,
        int num_combinations,
        unsigned char *combinations_array,
        int *end_indices,
        const CPUMatrix &measurements
    )
{
    cublasHandle_t handle;
    int block_size = 128;
    int info;
    int num_hypothesis = pow(num_buildingblocks, D) * num_combinations;
    int hypothesis_per_combination = num_hypothesis / num_combinations;
    int *dev_info_array;
    float *amatrices, *cmatrices, **amptrs, **cmptrs, *rss_costs, *smape_costs;
    GPUMatrix device_measurements = matrix_alloc_gpu(measurements.width, measurements.height);

    // download pointers for testing
    CPUMatrix A = matrix_alloc_cpu(measurements.height, D + 1);
    CPUMatrix X = matrix_alloc_cpu(1, D + 1);
    CPUMatrix C = matrix_alloc_cpu(1, measurements.height);

    // allocate and upload data
    matrix_upload(measurements, device_measurements);
    CUDA_CALL(cudaMemcpyToSymbol(combinations, combinations_array, num_combinations * D * D, 0, cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMalloc(&rss_costs, num_hypothesis * sizeof(float)));
    CUDA_CALL(cudaMalloc(&smape_costs, num_hypothesis * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dev_info_array, num_hypothesis * sizeof(int)));
    CUDA_CALL(cudaMalloc(&amatrices, num_hypothesis * measurements.height * (D+1) * sizeof(float)))
    CUDA_CALL(cudaMalloc(&cmatrices, num_hypothesis * measurements.height * sizeof(float)))
    CUDA_CALL(cudaMalloc(&amptrs, num_hypothesis * sizeof(float*)))
    CUDA_CALL(cudaMalloc(&cmptrs, num_hypothesis * sizeof(float*)))
    CUDA_CALL(cudaMemset(rss_costs, 0, num_hypothesis * sizeof(float)))
    CUDA_CALL(cudaMemset(smape_costs, 0, num_hypothesis * sizeof(float)))

    CUBLAS_CALL(cublasCreate(&handle));

    for (int i = 0; i < measurements.height - 1; i++) {
        prepare_gels_batched<D><<<div_up(num_hypothesis, block_size), block_size>>>(
                device_measurements,
                num_combinations,
                num_buildingblocks,
                num_hypothesis,
                amatrices,
                cmatrices,
                amptrs,
                cmptrs,
                i
        );

        int start_index = 0;
        for (int j = 0; j < D; j++) {
            int end_index = end_indices[j];
            if (end_index == -1) continue;
            int combination_count = end_index - start_index;
            CUBLAS_CALL(cublasSgelsBatched(
                    handle,
                    CUBLAS_OP_N,
                    measurements.height - 1, // height of Aarray
                    j + 2, // width of Aarray and height of Carray
                    1, // width of Carray
                    amptrs + (hypothesis_per_combination * start_index), // Aarray pointer
                    measurements.height, // lda >= max(1,m)
                    cmptrs + (hypothesis_per_combination * start_index), // Carray pointer
                    measurements.height, // ldc >= max(1,m)
                    &info,
                    dev_info_array,
                    combination_count * hypothesis_per_combination
                )
            )
            start_index = end_index;
        }

        compute_costs<D><<<div_up(num_hypothesis, 512), 512>>>(device_measurements, num_combinations, num_buildingblocks,
                                                               num_hypothesis, cmatrices, rss_costs, smape_costs, i);
    }

    prepare_gels_batched<D><<<div_up(num_hypothesis, block_size), block_size>>>(
            device_measurements,
            num_combinations,
            num_buildingblocks,
            num_hypothesis,
            amatrices,
            cmatrices,
            amptrs,
            cmptrs,
            measurements.width - 1
    );

    int start_index = 0;
    for (int i = 0; i < D; i++) {
        int end_index = end_indices[i];
        if (end_index == -1) continue;
        int combination_count = end_index - start_index;
        CUBLAS_CALL(cublasSgelsBatched(
                handle,
                CUBLAS_OP_N,
                measurements.height, // height of Aarray
                i + 2, // width of Aarray and height of Carray
                1, // width of Carray
                amptrs + (hypothesis_per_combination * start_index), // Aarray pointer
                measurements.height, // lda >= max(1,m)
                cmptrs + (hypothesis_per_combination * start_index), // Carray pointer
                measurements.height, // ldc >= max(1,m)
                &info,
                dev_info_array,
                combination_count * hypothesis_per_combination
            )
        )
        start_index = end_index;
    }

    int minimum_smape_cost;
    CUBLAS_CALL(cublasIsamin_v2(handle, num_hypothesis, smape_costs, 1, &minimum_smape_cost));
    minimum_smape_cost -= 1;
    printf("SMAPE cost index: %d\n", minimum_smape_cost);
    print_hypothesis<D><<<1, 1>>>(minimum_smape_cost, num_combinations, num_buildingblocks, num_hypothesis, device_measurements, cmatrices, smape_costs);

    CUDA_CALL(cudaDeviceSynchronize());

    matrix_free_cpu(A);
    matrix_free_cpu(X);
    matrix_free_cpu(C);
    CUDA_CALL(cudaFree(amatrices))
    CUDA_CALL(cudaFree(cmatrices))
    CUDA_CALL(cudaFree(amptrs))
    CUDA_CALL(cudaFree(cmptrs))
    CUDA_CALL(cudaFree(rss_costs))
    CUDA_CALL(cudaFree(smape_costs))
    CUDA_CALL(cudaFree(dev_info_array))
    matrix_free_gpu(device_measurements);
}

void find_hypothesis(const CPUMatrix &measurements) {
    cublasHandle_t handle;
    int num_combinations;
    int num_buildingblocks = 43;
    int dimensions = measurements.width-1;
    switch(dimensions) {
        case 2:
            num_combinations = 4;
            find_hypothesis_templated<2>(
                    num_buildingblocks,
                    num_combinations,
                    combinations_2d,
                    combinations_2d_end_indices,
                    measurements
            );
            break;
        case 3:
            num_combinations = 23;
            find_hypothesis_templated<3>(
                    num_buildingblocks,
                    num_combinations,
                    combinations_3d,
                    combinations_3d_end_indices,
                    measurements
            );

            break;
        case 4:

            break;
        case 5:

            break;

        default:
            std::cerr << "Finding hypothesis with dimensions " << dimensions << " is not supported!" << std::endl;
            exit(EXIT_FAILURE);
    }

    // TODO return the hypothesis
}
