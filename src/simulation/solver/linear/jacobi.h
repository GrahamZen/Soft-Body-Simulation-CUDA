#pragma once

#include <linear/linear.h>
#include <cusolverSp.h>


template<typename T>
class JacobiSolver : public LinearSolver<T> {
public:
    JacobiSolver(int N, int maxIter = 1000);
    virtual ~JacobiSolver() override;
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    int maxIter;
    cusparseHandle_t cusHandle = nullptr;
    T* x_prime = nullptr;
};