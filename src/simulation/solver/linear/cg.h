#pragma once

#include <linear/linear.h>
#include <cusparse.h>
#include <cublas_v2.h>

template<typename T>
class CGSolver : public LinearSolver<T> {
public:
    CGSolver(int N, int max_iter = 1e2, T tolerance = 1e-6);
    virtual ~CGSolver() override;
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A, int nz, int* d_rowIdx, int* d_colIdx, T* d_guess = nullptr) override;
private:
    using LinearSolver<T>::dType;
    using LinearSolver<T>::d_A;
    using LinearSolver<T>::d_rowIdx;
    using LinearSolver<T>::d_colIdx;
    using LinearSolver<T>::d_rowPtrA;
    using LinearSolver<T>::capacity;

    cublasHandle_t cubHandle = nullptr;
    cusparseHandle_t cusHandle = nullptr;
    csric02Info_t ic02info = nullptr;

    cusparseMatDescr_t descrA = nullptr;
    cusparseMatDescr_t descrL = nullptr;
    cusparseDnVecDescr_t dvec_x = nullptr, dvec_b = nullptr, dvec_r = nullptr, dvec_y = nullptr, dvec_z = nullptr, dvec_p = nullptr, dvec_q = nullptr;
    cusparseSpMatDescr_t d_matA = nullptr, d_matL = nullptr;
    cusparseSpSVDescr_t spsvDescrL = nullptr;
    cusparseSpSVDescr_t spsvDescrU = nullptr;

    int N = 0;
    int max_iter;
    int k = 0;  // k iteration
    T tolerance;
    T alpha = 0;
    T beta = 0;
    T rTr = 0;
    T pTq = 0;
    T rho = 0;    //rho{k}
    T rho_t = 0;  //rho{k-1}
    const T one = 1.0;  // constant
    const T zero = 0.0;   // constant

    T* d_ic = nullptr;  // Factorized L
    T* d_y = nullptr;
    T* d_z = nullptr;
    T* d_r = nullptr;
    T* d_q = nullptr;
    T* d_p = nullptr;
    void* d_bufL = nullptr;
    void* d_bufU = nullptr;
    void* ic02Buffer = nullptr;

    // old memory size
    size_t old_nnz = 0;
    size_t old_ic02BufferSizeInBytes = 0;
    size_t old_bufferSizeL = 0;
    size_t old_bufferSizeU = 0;
};