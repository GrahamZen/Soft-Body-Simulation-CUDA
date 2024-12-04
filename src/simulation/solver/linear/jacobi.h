#pragma once

#include <linear/linear.h>
#include <cusolverSp.h>


template<typename T>
class JacobiSolver : public LinearSolver<T> {
    cudaDataType dType = CUDAType<T>::value;
public:
    JacobiSolver(int N, int maxIter = 1000);
    virtual ~JacobiSolver() override;
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    int maxIter;
    void* d_buf = nullptr;
    T* d_r = nullptr;
    cublasHandle_t cubHandle = nullptr;
    cusparseHandle_t cusHandle = nullptr;
    cusparseDnVecDescr_t dvec_x = nullptr, dvec_r = nullptr;
    cusparseSpMatDescr_t d_matA = nullptr;
    const T one = 1.0;  // constant
    const T negone = -1.0;  // constant
    const T zero = 1.0;  // constant
};