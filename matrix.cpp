#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>

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
    CUDA_CALL(cudaMallocPitch(&mat.elements, &mat.pitch, width * sizeof(float), height))
    return mat;
}

void matrix_free_gpu(GPUMatrix &m)
{
    CUDA_CALL(cudaFree(m.elements))
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
    CUDA_CALL(cudaMemcpy2D(
            dst.elements,
            dst.pitch,
            src.elements,
            src.width * sizeof(float),
            src.width * sizeof(float),
            src.height,
            cudaMemcpyHostToDevice
        )
    )
}
void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
    CUDA_CALL(cudaMemcpy2D(
            dst.elements,
            dst.width * sizeof(float),
            src.elements,
            src.pitch,
            src.width * sizeof(float),
            src.height,
            cudaMemcpyDeviceToHost
        )
    )
}

void matrix_print(const CPUMatrix &m) {
    std::cout << "[" << std::endl;
    for (int y = 0; y < m.height; y++) {
        std::cout << "[";
        for (int x = 0; x < m.width; x++) {
            std::cout << m.elements[y * m.width + x] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]" << std::endl;
}

CPUMatrix load_from_file(const std::string& filename) {
    std::cout << "Opening file '" << filename << "'." << std::endl;

    std::ifstream file;
    file.open(filename);
    if(!file.is_open()) {
        std::cerr << "Error: could not load file '" << filename << "'!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string extrapStr, measurementsStr;
    file >> extrapStr >> measurementsStr;

    if(extrapStr != "extrap" || measurementsStr != "measurements") {
        std::cerr << "Error: file '" << filename << "' is not an extrap measurements file!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dimensions, rows;
    file >> dimensions >> rows;

    std::cout << rows << " measurements with " << dimensions << " dimensions found." << std::endl;

    CPUMatrix measurements = matrix_alloc_cpu(dimensions+1, rows);

    for(int row = 0; row < rows; row++) {
        float num;
        for(int i = 0; i < dimensions+1; i++) {
            file >> num;
            measurements.elements[row*(dimensions + 1) + i] = num;
        }
    }

    file.close();

    std::cout << "Measurements successfully loaded." << std::endl;

    return measurements;
}

CPUMatrix row_randomized_copy(const CPUMatrix &m) {
    std::vector<int> indices;
    indices.resize(m.height);
    for (int y = 0; y < m.height; y++) {
        indices[y] = y;
    }
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
    CPUMatrix r = matrix_alloc_cpu(m.width, m.height);
    for (int y = 0; y < m.height; y++) {
        for (int x = 0; x < m.width; x++) {
            r.elements[y * m.width + x] = m.elements[indices[y] * m.width + x];
        }
    }
    return r;
}
