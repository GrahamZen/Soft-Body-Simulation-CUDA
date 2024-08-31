#pragma once
#include <def.h>

template<typename T>
class LinearSolver {
public:
    LinearSolver() = default;
    virtual ~LinearSolver() = default;
    virtual void Solve(int N, T* d_b, T* d_x, T* A = nullptr, int nz = 0, int* rowIdx = nullptr, int* colIdx = nullptr, T* d_guess = nullptr) = 0;
protected:
    T* d_A = nullptr;
    int* d_rowIdx = nullptr;
    int* d_colIdx = nullptr;
    int* d_rowPtrA = nullptr; // CSR 
};
