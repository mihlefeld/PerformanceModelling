#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>

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
    return *get_matrix_element_ptr(m, x, y);
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
__global__ void __launch_bounds__(256)
prepare_gels_batched(GPUMatrix measurements, Counts counts, Matrices mats, int offset, int tcount = 0) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= counts.batch_size || idx + offset >= counts.hypotheses)
        return;

    float *A = &mats.A[idx * (measurements.height * (D + 1))];
    float *C = &mats.C[idx * measurements.height];
    mats.aps[idx] = A;
    mats.cps[idx] = C;
    // TODO: this really shouldn't be here, I don't even know why it's faster tbh
    __shared__ float sctps[512][2*D];
    float *ctps = sctps[threadIdx.x];
    unsigned char *combination;

    get_data_from_indx<D>(offset + idx, ctps, &combination, counts);

    for (int i = 0; i < measurements.height - tcount; i++) {
        // first element in every row should be 1, since there's always a constant component
        A[i] = 1;
    }

    for (int i = 0; i < measurements.height - tcount; i++) {
        // TODO: seperate coordinates and values
        C[i] = get_matrix_element(measurements, D, i);
    }

    // TODO: alternative kernel for more than 500 measurements
    float column_cache[500];
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < measurements.height - tcount; i++) {
            column_cache[i] = evaluate_single<D>(&combination[D*j], 1, ctps, get_matrix_element_ptr(measurements, 0, i));
        }
        for (int i = 0; i < measurements.height - tcount; i++) {
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

template <int D>
__global__ void save_hypothesis(GPUHypothesis g_hypo, int idx, int offset, Counts counts, GPUMatrix measurements, Matrices mats, Costs costs) {
    float *coefs = &mats.C[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx + offset, g_hypo.exponents, &combination, counts);
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
void solve(CublasStuff cbstuff, Counts counts, Matrices mats, int offset, const int *end_indices, int solve_count) {
    int start_index = 0;
    for (int j = 0; j < D; j++) {
        int end_index = end_indices[j] * counts.hpc;
        int osi = max(0, start_index - offset);
        int oei = min(counts.batch_size - 1, end_index - offset);
        int occ = oei - osi;
        if (occ <= 0) continue;
        CUBLAS_CALL(cublasSgelsBatched(
                cbstuff.handle,
                CUBLAS_OP_N,
                solve_count, // height of Aarray
                j + 2, // width of Aarray and height of Carray
                1, // width of Carray
                mats.aps + osi, // Aarray pointer
                cbstuff.lda, // lda >= max(1,m)
                mats.cps + osi, // Carray pointer
                cbstuff.lda, // ldc >= max(1,m)
                &cbstuff.info,
                nullptr,
                occ
        )
        )
        start_index = end_index;
    }
}

__global__ void segment_training_data(GPUMatrix src_measurements, GPUMatrix dst_measurements, int start, int end, int size) {
    int idx = threadIdx.x;
    int h = src_measurements.height;
    for (int i = 0; i < start; i++) {
        *get_matrix_element_ptr(dst_measurements, idx, i) = get_matrix_element(src_measurements, idx, i);
    }
    for (int i = start; i < end; i++) {
        float *loc = get_matrix_element_ptr(dst_measurements, idx, h + (i - start) - size);
        *loc = get_matrix_element(src_measurements, idx, i);
    }
    for (int i = end; i < h; i++) {
        float *loc = get_matrix_element_ptr(dst_measurements, idx, i - size);
        *loc = get_matrix_element(src_measurements, idx, i);
    }
}

template<int D>
__global__ void compute_fold_costs(GPUMatrix measurements, Counts counts, Matrices mats, Costs costs, int vs_size, int offset) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= counts.batch_size || idx + offset >= counts.hypotheses)
        return;

    float ctps[2*D];
    float *coefs = &mats.C[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx + offset, ctps, &combination, counts);

    float cur_rss = 0;
    float cur_smape = 0;
    for (int i = counts.measurements - vs_size; i < counts.measurements; i++) {
        float *row_ptr = get_matrix_element_ptr(measurements, 0, i);
        float actual = row_ptr[D];
        float predicted = evaluate_multi<D>(combination, coefs, ctps, row_ptr);
        cur_rss += rss(predicted, actual);
        cur_smape += smape(predicted, actual);
    }

    costs.rss[idx] += cur_rss;
    costs.smape[idx] += cur_smape / counts.measurements;
}

template<int D>
__global__ void compute_full_cost(GPUMatrix measurements, Counts counts, Matrices mats, Costs costs, int idx, int offset) {
    if(idx >= counts.batch_size || idx + offset >= counts.hypotheses)
        return;

    float ctps[2*D];
    float *coefs = &mats.C[idx * measurements.height];
    unsigned char *combination;
    get_data_from_indx<D>(idx + offset, ctps, &combination, counts);

    float cur_rss = 0;
    float cur_smape = 0;
    for (int i = 0; i < counts.measurements; i++) {
        float *row_ptr = get_matrix_element_ptr(measurements, 0, i);
        float actual = row_ptr[D];
        float predicted = evaluate_multi<D>(combination, coefs, ctps, row_ptr);
        cur_rss += rss(predicted, actual);
        cur_smape += smape(predicted, actual);
    }

    costs.rss[idx] = cur_rss;
    costs.smape[idx] = cur_smape / counts.measurements;
}

template<int D>
CPUHypothesis find_hypothesis_templated(
        Counts counts,
        unsigned char *combinations_array,
        int *end_indices,
        const CPUMatrix &measurements
    )
{
    // calculate fold sizes
    int k_folds = 5;
    int fold_size = counts.measurements / k_folds;
    float tfs = counts.measurements / (float) k_folds;
    int fold_sizes[k_folds];
    int total = 0;
    for (int i = 0; i < k_folds; i++) {
        float ideal = (i + 1) * tfs;
        int cfs = total + fold_size < ideal ? fold_size + 1 : fold_size;
        total += cfs;
        fold_sizes[i] = cfs;
    }

    int block_size = 128;
    int grid_size = div_up(counts.batch_size, block_size);
    CublasStuff cbstuff = create_cublas_stuff(counts);
    Matrices mats = create_matrices(counts);
    Costs costs = create_costs(counts);
    CPUMatrix r_measurements = row_randomized_copy(measurements);
    GPUMatrix d_measurements = matrix_alloc_gpu(r_measurements.width, r_measurements.height);
    GPUMatrix tmp_measurements = matrix_alloc_gpu(r_measurements.width, r_measurements.height);
    std::vector<GPUHypothesis> d_hypotheses;
    std::vector<CPUHypothesis> c_hypotheses;
    matrix_upload(r_measurements, d_measurements);
    matrix_upload(r_measurements, tmp_measurements);
    CUDA_CALL(cudaMemcpyToSymbol(combinations, combinations_array, counts.combinations * D * D, 0, cudaMemcpyHostToDevice))


    for (int batch = 0; batch < counts.batches; batch++) {
        d_hypotheses.push_back(create_gpu_hypothesis(D));
        c_hypotheses.push_back(create_cpu_hypothesis(D));
    }

    for (int batch = 0; batch < counts.batches; batch++) {
        int offset = batch * counts.batch_size;
        int vs_start = 0;
        for (int i = 0; i < k_folds; i++) {
            int vs_size = fold_sizes[i];
            int vs_end = vs_start + vs_size;
            segment_training_data<<<1, D+1>>>(d_measurements, tmp_measurements, vs_start, vs_end, vs_size);
            prepare_gels_batched<D><<<grid_size, block_size>>>(tmp_measurements, counts, mats, offset, vs_size);
            solve<D>(cbstuff, counts, mats, offset, end_indices, r_measurements.height - vs_size);
            compute_fold_costs<D><<<grid_size, block_size>>>(tmp_measurements, counts, mats, costs, vs_size, offset);
            vs_start = vs_end;
        }

        prepare_gels_batched<D><<<grid_size, block_size>>>(d_measurements, counts, mats, offset);
        solve<D>(cbstuff, counts, mats, offset, end_indices, r_measurements.height);

        int min_cost_idx;
        CUBLAS_CALL(cublasIsamin_v2(cbstuff.handle, counts.batch_size, costs.smape, 1, &min_cost_idx))
        min_cost_idx -= 1;
        compute_full_cost<D><<<1, 1>>>(d_measurements, counts, mats, costs, min_cost_idx, offset);
        save_hypothesis<D><<<1, 1>>>(d_hypotheses[batch], min_cost_idx, offset, counts, d_measurements, mats, costs);

    }

    cudaDeviceSynchronize();

    CPUHypothesis best_hypothesis{};
    float min_smape = 300;
    for (int batch = 0; batch < counts.batches; batch++) {
        auto &cur = c_hypotheses[batch];
        cur.download(d_hypotheses[batch]);
        destroy_gpu_hypothesis(d_hypotheses[batch]);
        if (cur.smape < min_smape) {
            best_hypothesis = cur;
            min_smape = cur.smape;
        }
    }

    destroy_costs(costs);
    destroy_matrices(mats);
    destroy_cublas_stuff(cbstuff);
    matrix_free_gpu(d_measurements);
    matrix_free_gpu(tmp_measurements);
    matrix_free_cpu(r_measurements);

    return best_hypothesis;
}

CPUHypothesis find_hypothesis(const CPUMatrix &measurements) {
    CPUHypothesis best_hypothesis{};
    Counts counts{};
    int num_buildingblocks = 43;
    int dimensions = measurements.width-1;
    switch(dimensions) {
        case 2:
            counts = Counts(2, num_buildingblocks, 4, measurements.height);
            best_hypothesis = find_hypothesis_templated<2>(
                    counts,
                    combinations_2d,
                    combinations_2d_end_indices,
                    measurements
            );
            break;
        case 3:
            counts = Counts(3, num_buildingblocks, 23, measurements.height);
            best_hypothesis = find_hypothesis_templated<3>(
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

    best_hypothesis.print();

    return best_hypothesis;
}

CublasStuff create_cublas_stuff(Counts counts) {
    CublasStuff cbstuff{};
    cbstuff.lda = counts.measurements;
    CUBLAS_CALL(cublasCreate_v2(&cbstuff.handle));
    return cbstuff;
}

Matrices create_matrices(Counts counts) {
    Matrices mats{};
    CUDA_CALL(cudaMalloc(&mats.A, counts.batch_size * counts.measurements * (counts.dim+1) * sizeof(float)))
    CUDA_CALL(cudaMalloc(&mats.C, counts.batch_size * counts.measurements * sizeof(float)))
    CUDA_CALL(cudaMalloc(&mats.aps, counts.batch_size * sizeof(float*)))
    CUDA_CALL(cudaMalloc(&mats.cps, counts.batch_size * sizeof(float*)))
    return mats;
}

Costs create_costs(Counts counts) {
    Costs costs{};
    CUDA_CALL(cudaMalloc(&costs.rss, counts.batch_size * sizeof(float)))
    CUDA_CALL(cudaMalloc(&costs.smape, counts.batch_size * sizeof(float)))
    CUDA_CALL(cudaMemset(costs.rss, 0, counts.batch_size * sizeof(float)))
    CUDA_CALL(cudaMemset(costs.smape, 0, counts.batch_size * sizeof(float)))
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
    return hypo;
}

void destroy_gpu_hypothesis(GPUHypothesis g_hypo) {
    CUDA_CALL(cudaFree(g_hypo.combination))
    CUDA_CALL(cudaFree(g_hypo.coefficients))
    CUDA_CALL(cudaFree(g_hypo.exponents))
    CUDA_CALL(cudaFree(g_hypo.smape))
    CUDA_CALL(cudaFree(g_hypo.rss))
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
    size_t vram_target = device_props.totalGlobalMem * 0.9;
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
        if (i%d == d-1)
            std::cout << std::endl;
    }
    std::cout << "-----------------------------------------------------------------" << std::endl;
}
