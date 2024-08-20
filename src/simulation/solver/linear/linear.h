#pragma once
#include <def.h>
class LinearSolver {
public:
    LinearSolver() = default;
    virtual ~LinearSolver() = default;
    virtual void Solve(int N, float *d_b, float *d_x, float *d_A = nullptr, int nz = 0, int *d_rowIdx = nullptr, int *d_colIdx = nullptr, float *d_guess = nullptr) = 0;
};
