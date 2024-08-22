#pragma once

#include <linear/linear.h>
#include <cusparse.h>
#include <cublas_v2.h>

class CGSolver : public LinearSolver<double> {
public:
    CGSolver(int N, int max_iter = 1e2, double tolerance = 1e-6);
    virtual ~CGSolver() override;
    virtual void Solve(int N, double* d_b, double* d_x, double* d_A, int nz, int* d_rowIdx, int* d_colIdx, double* d_guess = nullptr) override;
private:
    cublasHandle_t cubHandle = nullptr;
    cusparseHandle_t cusHandle = nullptr;
    csric02Info_t ic02info = nullptr;

    cusparseMatDescr_t descrA = nullptr;
    cusparseMatDescr_t descrL = nullptr;
    cusparseDnVecDescr_t dvec_x = nullptr, dvec_b = nullptr, dvec_r = nullptr, dvec_y = nullptr, dvec_z = nullptr, dvec_p = nullptr, dvec_q = nullptr;
    cusparseSpMatDescr_t d_matA = nullptr, d_matL = nullptr;
    cusparseSpSVDescr_t spsvDescrL = nullptr;
    cusparseSpSVDescr_t spsvDescrU = nullptr;
    int* d_rowPtrA; // CSR 

    int N = 0;
    int max_iter;
    int k = 0;  // k iteration
    double tolerance;
    double alpha = 0;
    double beta = 0;
    double rTr = 0;
    double pTq = 0;
    double rho = 0;    //rho{k}
    double rho_t = 0;  //rho{k-1}
    const double one = 1.0;  // constant
    const double zero = 0.0;   // constant

    double* d_ic = nullptr;  // Factorized L
    double* d_y = nullptr;
    double* d_z = nullptr;
    double* d_r = nullptr;
    double* d_q = nullptr;
    double* d_p = nullptr;
    void* d_bufL = nullptr;
    void* d_bufU = nullptr;
};