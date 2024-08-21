#pragma once
#include <def.h>

template<typename T>
class LinearSolver {
public:
    LinearSolver() = default;
    virtual ~LinearSolver() = default;
    virtual void Solve(int N, T *d_b, T *d_x, T *d_A = nullptr, int nz = 0, int *d_rowIdx = nullptr, int *d_colIdx = nullptr, T *d_guess = nullptr) = 0;
};
