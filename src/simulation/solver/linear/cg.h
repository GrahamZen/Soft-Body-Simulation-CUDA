#pragma once

#include <linear/linear.h>
#include <cusparse.h>
#include <cublas_v2.h>

class CGSolver : public LinearSolver {
public:
    CGSolver(int N);
    virtual ~CGSolver() override;
    virtual void Solve(int N, float* d_b, float* d_x, float* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, float* d_guess = nullptr) override;
private:
    cublasHandle_t cubHandle=nullptr;
    cusparseHandle_t cusHandle=nullptr;
    csric02Info_t ic02info = nullptr;

    cusparseMatDescr_t descrA = nullptr;
    cusparseMatDescr_t descrL = nullptr;
    cusparseDnVecDescr_t dvec_x = nullptr, dvec_b = nullptr, dvec_r = nullptr, dvec_y = nullptr, dvec_z= nullptr;
    cusparseSpMatDescr_t spMatDescrA = nullptr, spMatDescrL = nullptr;
    cusparseSpSVDescr_t spsvDescrL = nullptr;
    cusparseSpSVDescr_t spsvDescrU = nullptr;
    int* d_rowPtrA; // CSR 

    int N = 0;
    int max_iter = 1e4;
    int k = 0;  // k iteration
    float tolerance = 1e-5;
    float alpha = 0;
    float beta = 0;
    float rTr = 0;
    float pTq = 0;
    float rho = 0;    //rho{k}
    float rho_t = 0;  //rho{k-1}
    const float one = 1.0;  // constant
    const float zero = 0.0;   // constant

    float* d_ic = nullptr;  // Factorized L
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    float* d_r = nullptr;
    float* d_rt = nullptr;
    float* d_xt = nullptr;
    float* d_q = nullptr;
    float* d_p = nullptr;
    void* d_buf = nullptr;
    void* d_buf1 = nullptr;
};