#pragma once

#include "cublas_v2.h"
#include "matrix.h"

struct Counts {
    int hypotheses;
    int batch_size;
    int batches;
    int measurements;
    int dim;
    int combinations;
    int building_blocks;
    int hpc;
    Counts() = default;
    Counts(int dim, int building_blocks, int combinations, int measurements);
};

struct Matrices {
    float **aps, **cps, *A, *C;
};

struct Costs {
    float *rss, *smape;
};

struct CublasStuff {
    cublasHandle_t handle;
    int info;
    int lda;
};


struct GPUHypothesis {
    int d;
    float *coefficients;
    float *exponents;
    float *smape;
    float *rss;
    unsigned char *combination;
};

struct CPUHypothesis {
    int d;
    float *coefficients;
    float *exponents;
    float smape;
    float rss;
    unsigned char *combination;
    void download(GPUHypothesis g_hypo);
    void print();
};

template<int D>
void find_hypothesis_templated(
        Counts counts,
        unsigned char *combinations_array,
        int *end_indices,
        const CPUMatrix &measurements
);

extern "C" void find_hypothesis(const CPUMatrix &measurements);

template<int D>
void solve(CublasStuff cbstuff, Counts counts, Matrices mats, int offset, const int *end_indices, int solve_count);

size_t calculate_memory_usage(Counts counts);

CublasStuff create_cublas_stuff(Counts counts);
Matrices create_matrices(Counts counts);
Costs create_costs(Counts counts);
GPUHypothesis create_gpu_hypothesis(int d);
CPUHypothesis create_cpu_hypothesis(int d);

void destroy_cublas_stuff(CublasStuff cbstuff);
void destroy_matrices(Matrices mats);
void destroy_costs(Costs costs);

void destroy_gpu_hypothesis(GPUHypothesis g_hypo);
void destroy_cpu_hypothesis(CPUHypothesis c_hypo);