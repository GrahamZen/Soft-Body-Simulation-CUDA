#pragma once

#include <cuda_runtime.h>

class InertiaEnergy {
public:
    InertiaEnergy(int numVerts,const double* dev_mass);
    double Val(const double* dev_x, const double* dev_xTilde, const double* mass, int numVerts) const;
    void Gradient(double* grad, const double* dev_x, const double* dev_xTilde,const double* dev_mass, int numVerts) const;
    void Hessian(double*& hessianVal,  int*& hessianRowIdx,  int*& hessianColIdx) const;
private:
    int numVerts = 0;
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
};