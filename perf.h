#pragma once

#include "cublas_v2.h"
#include "matrix.h"

struct Counts {
    int hypotheses;
    int combinations;
    int building_blocks;
    int hpc;
    Counts() = default;
    Counts(int dim, int building_blocks, int combinations);
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

template<int D>
void find_hypothesis_templated(
        Counts counts,
        unsigned char *combinations_array,
        int *end_indices,
        const CPUMatrix &measurements
);

void find_hypothesis(const CPUMatrix &measurements);

template<int D>
void solve(CublasStuff cbstuff, Counts counts, Matrices mats, const int *end_indices, int solve_count);

CublasStuff create_cublas_stuff(int lda);
Matrices create_matrices(Counts counts, CPUMatrix measurements, int D);
Costs create_costs(Counts counts);

void destroy_cublas_stuff(CublasStuff cbstuff);
void destroy_matrices(Matrices mats);
void destroy_costs(Costs costs);