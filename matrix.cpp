#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

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

std::pair<CPUMatrix, CPUMatrix> load_from_file(const std::string& filename) {
    std::ifstream file;
    file.open(filename);
    if(!file.is_open()) {
        std::cerr << "Error: could not load file '" << filename << "'" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string extrapStr, measurementsStr;
    file >> extrapStr >> measurementsStr;

    if(extrapStr != "extrap" || measurementsStr != "measurements") {
        std::cerr << "Error: file '" << filename << "' is not an extrap measurements file" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dimensions, count;
    file >> dimensions >> count;

    std::cout << dimensions << " dimensions and " << count << " rows" << std::endl;

    CPUMatrix coordinates = matrix_alloc_cpu(dimensions, count);
    CPUMatrix measurements = matrix_alloc_cpu(1, count);

    for(int row = 0; row < count; row++) {
        float num;
        for(int i = 0; i < dimensions; i++) {
            file >> num;
            coordinates.elements[row*dimensions + i] = num;
        }

        file >> num;
        measurements.elements[row] = num;
    }

    return std::make_pair(coordinates, measurements);
}
