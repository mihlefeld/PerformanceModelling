#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#include "cublas_v2.h"
#include "perf.h"
#include "common.h"

// Device list of all possible exponents i and j, 2 wide and num_building_blocks high, stored row major
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

/*
 * Device list of all possible combinations
 * combinations are D*D large and represent how the terms should be added and multiplied together
 * they are stored in row major format and for every row the spot in the row indicates which term it represents
 * when the element is 1 it is present as part of the multiplicative row, if the whole row is 0, then the result
 * will be 0. All D rows get added together, this allows us to represent all possible combinations.
 * Over every hard-coded combination the add/multiply meaning is written as a comment.
 */
__constant__ unsigned char combinations[256];

/*
 * The end_indices group the combinations by the number of 0 rows. This is needed because cublasSgelsBatched
 * can only solve systems with a matrix A of full rank, when a 0 row is present, the resulting A matrix has as many
 * 0 columns as 0 rows in the combination. So we need to seperate these possibilities.
 * The end index is always the first index that is no longer part of the group.
 */
int combinations_2d_end_indices[] {
    1, 4
};

unsigned char combinations_2d[] {
    // 1c: a*b
    1, 1,
    0, 0,

    // 2c: a + b
    0, 1,
    1, 0,

    // 2c: a*b + a
    1, 1,
    1, 0,

    // 3c: a*b + b
    1, 1,
    0, 1
};

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

// -1 indicates, that there are no combinations with that many 0 rows
int combinations_4d_end_indices[] {
    1, -1, -1, 2
};

unsigned char combinations_4d[] {
    // 1c: a*b*c*d
    1, 1, 1, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,

    // 4c: a + b + c + d
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
};

int combinations_5d_end_indices[] {
    1, -1, -1, -1, 2
};

unsigned char combinations_5d[] {
    // 1c: a*b*c*d*e
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    // 5c: a + b + c + d + e
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

template<int D>
__device__ float evaluate_single(unsigned char *combination, float coef, float *ctps, float *params) {
    float prod = coef;
    // if the combination is 0 0 0, zero should be returned, instead of prod
    bool nonzero = false;
    for (int i = 0; i < D; i++) {
        nonzero |= combination[i];
        if (combination[i])
            prod *= pow(params[i], ctps[i*2]) * pow(log2(params[i]), ctps[i*2 + 1]);
    }
    return nonzero ? prod : 0;
}

template<int D>
__device__ float evaluate_multi(unsigned char *combination, float *coefs, float *ctps, float *params) {
    float result = coefs[0];
    for (int i = 0; i < D; i++) {
        result += evaluate_single<D>(&combination[i*D], coefs[i + 1], ctps, params);;
    }
    return result;
}

template<int D>
__device__ void get_data_from_indx(int idx, float *ctps, unsigned char **combination, Counts counts) {
    int combination_index = idx / counts.hpc;
    int mod_idx = idx  % counts.hpc;
    *combination = &combinations[combination_index * D * D];

    int r = counts.hpc;
    for (int i = D - 1; i >= 0; i--) {
        r/= counts.building_blocks;
        int ctpi = mod_idx / r;
        mod_idx = mod_idx % r;
        for (int j = 0; j < 2; j++) {
            ctps[i * 2 + j] = building_blocks[ctpi * 2 + j];
        }
    }
}

template<int D>
__global__ void __launch_bounds__(256) prepare_gels_batched(GPUMatrix measurements, Counts counts, Matrices mats, int swap_indx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= counts.hypotheses)
        return;

    float *A = &mats.A[idx * (measurements.height * (D + 1))];
    float *C = &mats.C[idx * measurements.height];
    mats.aps[idx] = A;
    mats.cps[idx] = C;
    // TODO: this really shouldn't be here, I don't even know why it's faster tbh
    __shared__ float sctps[512][2*D];
    float *ctps = sctps[threadIdx.x];
    unsigned char *combination;

    get_data_from_indx<D>(idx, ctps, &combination, counts);

    for (int i = 0; i < measurements.height; i++) {
        // first element in every row should be 1, since there's always a constant component
        A[i] = 1;
    }

    for (int i = 0; i < measurements.height; i++) {
        // TODO: seperate coordinates and values
        int ii = i == swap_indx ? (measurements.height - 1) : (i == measurements.height - 1 ? swap_indx : i);
        C[i] = get_matrix_element(measurements, D, ii);
    }

    // TODO: alternative kernel for more than 500 measurements
    float column_cache[500];
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < measurements.height; i++) {
            int ii = i == swap_indx ? (measurements.height - 1) : (i == measurements.height - 1 ? swap_indx : i);
            column_cache[i] = evaluate_single<D>(&combination[D*j], 1, ctps, get_matrix_element_ptr(measurements, 0, ii));
        }
        for (int i = 0; i < measurements.height; i++) {
            A[(j + 1) * measurements.height + i] = column_cache[i];
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
__global__ void compute_costs(GPUMatrix measurements, Counts counts,
                              Matrices mats, Costs costs,
                              int validation_index) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= counts.hypotheses)
        return;
    float ctps[2*D];
    float *coefs = &mats.C[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx, ctps, &combination, counts);

    // assume the validation measurement is the last row in the matrix
    int i = validation_index;
    float *row_ptr = get_matrix_element_ptr(measurements, 0, i);
    float actual = row_ptr[D];
    float predicted = evaluate_multi<D>(combination, coefs, ctps, row_ptr);

    costs.rss[idx] += rss(predicted, actual);
    costs.smape[idx] += smape(predicted, actual) / (measurements.height - 1);
}

template <int D>
__global__ void save_hypothesis(GPUHypothesis g_hypo, int idx, Counts counts, GPUMatrix measurements, Matrices mats, Costs costs) {
    float *coefs = &mats.C[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx, g_hypo.exponents, &combination, counts);
    *g_hypo.smape = costs.smape[idx];
    *g_hypo.rss = costs.rss[idx];
    for (int i = 0; i < D*D; i++) {
        g_hypo.combination[i] = combination[i];
    }
    for (int i = 0; i < D + 1; i++) {
        g_hypo.coefficients[i] = coefs[i];
    }
}

template<int D>
void solve(CublasStuff cbstuff, Counts counts, Matrices mats, const int *end_indices, int solve_count) {
    int start_index = 0;
    for (int j = 0; j < D; j++) {
        int end_index = end_indices[j];
        if (end_index == -1) continue;
        int combination_count = end_index - start_index;
        CUBLAS_CALL(cublasSgelsBatched(
                cbstuff.handle,
                CUBLAS_OP_N,
                solve_count, // height of Aarray
                j + 2, // width of Aarray and height of Carray
                1, // width of Carray
                mats.aps + (counts.hpc * start_index), // Aarray pointer
                cbstuff.lda, // lda >= max(1,m)
                mats.cps + (counts.hpc * start_index), // Carray pointer
                cbstuff.lda, // ldc >= max(1,m)
                &cbstuff.info,
                nullptr,
                combination_count * counts.hpc
        )
        )
        start_index = end_index;
    }
}

template<int D>
void find_hypothesis_templated(
        Counts counts,
        unsigned char *combinations_array,
        int *end_indices,
        const CPUMatrix &measurements
    )
{
    int block_size = 128;
    int grid_size = div_up(counts.hypotheses, block_size);
    int info;
    CublasStuff cbstuff = create_cublas_stuff(counts);
    Matrices mats = create_matrices(counts);
    Costs costs = create_costs(counts);
    GPUMatrix d_measurements = matrix_alloc_gpu(measurements.width, measurements.height);
    matrix_upload(measurements, d_measurements);
    CUDA_CALL(cudaMemcpyToSymbol(combinations, combinations_array, counts.combinations * D * D, 0, cudaMemcpyHostToDevice))


    for (int i = 0; i < measurements.height - 1; i++) {
        prepare_gels_batched<D><<<grid_size, block_size>>>(d_measurements, counts, mats, i);

        solve<D>(cbstuff, counts, mats, end_indices, measurements.height - 1);

        compute_costs<D><<<grid_size, block_size>>>(d_measurements, counts, mats, costs, i);
    }

    prepare_gels_batched<D><<<grid_size, block_size>>>(d_measurements, counts,mats, measurements.height - 1);

    solve<D>(cbstuff, counts, mats, end_indices, measurements.height);

    int min_cost_idx;
    CUBLAS_CALL(cublasIsamin_v2(cbstuff.handle, counts.hypotheses, costs.smape, 1, &min_cost_idx));
    min_cost_idx -= 1;
    GPUHypothesis g_hypo = create_gpu_hypothesis(D);
    CPUHypothesis c_hypo = create_cpu_hypothesis(D);
    save_hypothesis<D><<<1, 1>>>(g_hypo, min_cost_idx, counts, d_measurements, mats, costs);
    c_hypo.download(g_hypo);
    c_hypo.print();

    CUDA_CALL(cudaDeviceSynchronize());

    destroy_costs(costs);
    destroy_matrices(mats);
    destroy_cublas_stuff(cbstuff);
    destroy_gpu_hypothesis(g_hypo);
    destroy_cpu_hypothseis(c_hypo);
    matrix_free_gpu(d_measurements);
}

void find_hypothesis(const CPUMatrix &measurements) {
    cublasHandle_t handle;
    Counts counts;
    int num_buildingblocks = 39;
    int dimensions = measurements.width-1;
    switch(dimensions) {
        case 2:
            counts = Counts(2, num_buildingblocks, 4, measurements.height);
            find_hypothesis_templated<2>(
                    counts,
                    combinations_2d,
                    combinations_2d_end_indices,
                    measurements
            );
            break;
        case 3:
            counts = Counts(3, num_buildingblocks, 23, measurements.height);
            find_hypothesis_templated<3>(
                    counts,
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

CublasStuff create_cublas_stuff(Counts counts) {
    CublasStuff cbstuff{};
    cbstuff.lda = counts.measurements;
    CUBLAS_CALL(cublasCreate_v2(&cbstuff.handle));
    return cbstuff;
}

Matrices create_matrices(Counts counts) {
    Matrices mats{};
    CUDA_CALL(cudaMalloc(&mats.A, counts.hypotheses * counts.measurements * (counts.dim+1) * sizeof(float)))
    CUDA_CALL(cudaMalloc(&mats.C, counts.hypotheses * counts.measurements * sizeof(float)))
    CUDA_CALL(cudaMalloc(&mats.aps, counts.hypotheses * sizeof(float*)))
    CUDA_CALL(cudaMalloc(&mats.cps, counts.hypotheses * sizeof(float*)))
    return mats;
}

Costs create_costs(Counts counts) {
    Costs costs{};
    CUDA_CALL(cudaMalloc(&costs.rss, counts.hypotheses * sizeof(float)))
    CUDA_CALL(cudaMalloc(&costs.smape, counts.hypotheses * sizeof(float)))
    CUDA_CALL(cudaMemset(costs.rss, 0, counts.hypotheses * sizeof(float)))
    CUDA_CALL(cudaMemset(costs.smape, 0, counts.hypotheses * sizeof(float)))
    return costs;
}

void destroy_cublas_stuff(CublasStuff cbstuff) {
    CUBLAS_CALL(cublasDestroy_v2(cbstuff.handle))
}

void destroy_matrices(Matrices mats) {
    CUDA_CALL(cudaFree(mats.A))
    CUDA_CALL(cudaFree(mats.C))
    CUDA_CALL(cudaFree(mats.aps))
    CUDA_CALL(cudaFree(mats.cps))
}

void destroy_costs(Costs costs) {
    CUDA_CALL(cudaFree(costs.rss))
    CUDA_CALL(cudaFree(costs.smape))
}

GPUHypothesis create_gpu_hypothesis(int d) {
    GPUHypothesis hypo{};
    hypo.d = d;
    CUDA_CALL(cudaMalloc(&hypo.combination, d * d))
    CUDA_CALL(cudaMalloc(&hypo.coefficients, (d + 1) * sizeof(float)))
    CUDA_CALL(cudaMalloc(&hypo.exponents, 2 * d * sizeof(float)))
    CUDA_CALL(cudaMalloc(&hypo.smape, sizeof(float)))
    CUDA_CALL(cudaMalloc(&hypo.rss, sizeof(float)))
    return hypo;
}

CPUHypothesis create_cpu_hypothesis(int d) {
    CPUHypothesis hypo{};
    hypo.d = d;
    hypo.combination = new unsigned char[d * d];
    hypo.coefficients = new float[d + 1];
    hypo.exponents = new float[2 * d];
    return hypo;
}

void destroy_gpu_hypothesis(GPUHypothesis g_hypo) {
    CUDA_CALL(cudaFree(g_hypo.combination))
    CUDA_CALL(cudaFree(g_hypo.coefficients))
    CUDA_CALL(cudaFree(g_hypo.exponents))
    CUDA_CALL(cudaFree(g_hypo.smape))
    CUDA_CALL(cudaFree(g_hypo.rss))
}

void destroy_cpu_hypothseis(CPUHypothesis c_hypo) {
    delete [] c_hypo.combination;
    delete [] c_hypo.coefficients;
    delete [] c_hypo.exponents;
}

size_t calculate_memory_usage(Counts counts) {
    size_t sof = sizeof(float);
    size_t size_costs = counts.hypotheses * 2 * sof;
    size_t size_mat_ptrs = counts.hypotheses * 2 * sizeof(float*);
    size_t size_a_matrix = counts.hypotheses * counts.measurements * (counts.dim + 1) * sof;
    size_t size_c_vector = counts.hypotheses * counts.measurements * sof;
    return size_costs + size_mat_ptrs + size_a_matrix + size_c_vector;
}

Counts::Counts(int dim, int building_blocks, int combinations, int measurements):
    dim(dim), building_blocks(building_blocks), combinations(combinations), measurements(measurements) {
    hpc = pow(building_blocks, dim);
    hypotheses = combinations * hpc;
    cudaDeviceProp device_props{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&device_props, device);
    size_t vram_target = device_props.totalGlobalMem * 0.8;
    size_t vram_cost = calculate_memory_usage(*this);
    batches = ceil(vram_cost / (float) vram_target);
    batch_size = ceil(hypotheses / (float) batches);
}

void CPUHypothesis::download(GPUHypothesis g_hypo) {
    CUDA_CALL(cudaMemcpy(combination, g_hypo.combination, g_hypo.d * g_hypo.d, cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(coefficients, g_hypo.coefficients, (g_hypo.d + 1)*sizeof(float), cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(exponents, g_hypo.exponents, g_hypo.d*2*sizeof(float), cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(&smape, g_hypo.smape, sizeof(float), cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(&rss, g_hypo.rss, sizeof(float), cudaMemcpyDeviceToHost))
}

void CPUHypothesis::print() {
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Hypothesis (SMAPE = " << smape << ", RSS = " << rss << ")" << std::endl;
    std::cout << "Coefficients:";
    for (int i = 0; i < d + 1; i++)
        std::cout << " " << coefficients[i];
    std::cout << "\nExponents:";
    for (int i = 0; i < 2*d; i+=2) {
        std::cout << " " << "(" << exponents[i] << ", " << exponents[i+1] << ")";
    }
    std::cout << "\nCombination:" << std::endl;
    for (int i = 0; i < d * d; i++) {
        std::cout << (int) combination[i] << ", ";
        if (i%d == 1)
            std::cout << std::endl;
    }
    std::cout << "-----------------------------------------------------------------" << std::endl;
}
