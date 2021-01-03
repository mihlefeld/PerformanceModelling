#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"

CPUMatrix matrix_alloc_cpu(int width, int height)
{
    CPUMatrix m{};
    m.width = width;
    m.height = height;
    m.elements = new float[m.width * m.height];
    return m;
}

void matrix_free_cpu(CPUMatrix &m)
{
    delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
    GPUMatrix mat{};
    mat.width = width;
    mat.height = height;
    cudaMallocPitch(&mat.elements, &mat.pitch, width * sizeof(float), height);
    CUDA_CHECK_ERROR;
    return mat;
}

void matrix_free_gpu(GPUMatrix &m)
{
    cudaFree(m.elements);
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
    cudaMemcpy2D(dst.elements, dst.pitch, src.elements, src.width * sizeof(float),
                 src.width * sizeof(float), src.height,
                 cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}
void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
    cudaMemcpy2D(dst.elements, dst.width * sizeof(float), src.elements, src.pitch,
                 src.width * sizeof(float), src.height,
                 cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR;
}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b)
{
    for (int y = 0; y < a.height; y++) {
        for (int x = 0; x < a.width; x++) {
            std::cout << std::abs(a.elements[y * a.width + x] - b.elements[y * b.width + x]) << " ";
        }
        std::cout << std::endl;
    }
}

void matrix_print(const CPUMatrix &m) {
    for (int y = 0; y < m.height; y++) {
        for (int x = 0; x < m.width; x++) {
            std::cout << m.elements[y * m.width + x] << " ";
        }
        std::cout << std::endl;
    }
}