#pragma once

#include <linear/linear.h>
#include <cusparse.h>
#include <cublas_v2.h>

template<typename T>
class PCGJacobiSolver : public LinearSolver<T> {
public:
    PCGJacobiSolver(int N, int max_iter = 2000, T tolerance = 1e-5);
    virtual ~PCGJacobiSolver() override;
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

    cusparseMatDescr_t descrA = nullptr;
    
    // Dense vectors
    cusparseDnVecDescr_t dvec_x = nullptr, dvec_b = nullptr, dvec_r = nullptr;
    cusparseDnVecDescr_t dvec_p = nullptr, dvec_q = nullptr;
    
    // Matrix descriptors
    cusparseSpMatDescr_t d_matA = nullptr; 

    int N = 0;
    int max_iter;
    int k = 0;  // k iteration
    T tolerance;
    T alpha = 0;
    T beta = 0;
    T rTr = 0;
    T pTq = 0;
    T rho = 0;
    T rho_t = 0;
    const T one = 1.0;
    const T zero = 0.0;

    // Buffers
    T* d_z = nullptr; // Preconditioned residual
    T* d_r = nullptr;
    T* d_q = nullptr;
    T* d_p = nullptr;
    T* d_x = nullptr;
    T* d_b = nullptr;
    
    // Jacobi Preconditioner: inverse of diagonal elements
    T* d_invDiag = nullptr; 

    void* d_bufMV = nullptr;
    size_t old_bufferSizeMV = 0;
};