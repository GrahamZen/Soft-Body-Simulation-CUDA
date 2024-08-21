#include <linear/cg.h>
#include <linear/error_helper.h>
#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
template <typename T>
void inspectHost(const T* host_ptr, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;

    for (int i = 0; i < size; i++) {
        std::cout << host_ptr[i] << std::endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}


template <typename T>
void inspectGLM(T* dev_ptr, int size) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    inspectHost(host_ptr.data(), size);
}
void printSparseCOOToFull(int N, int nz, float* d_val, int* d_rowIdx, int* d_colIdx) {
    std::vector<int> rowIdx(nz);
    std::vector<int> colIdx(nz);
    std::vector<float> val(nz);

    cudaMemcpy(rowIdx.data(), d_rowIdx, sizeof(int) * nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(colIdx.data(), d_colIdx, sizeof(int) * nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(val.data(), d_val, sizeof(float) * nz, cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> full(N, std::vector<float>(N, 0));

    for (int i = 0; i < nz; i++) {
        full[rowIdx[i]][colIdx[i]] = val[i];
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << full[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
void printSparseCSRToFull(int N, int nz, float* d_val, int* d_rowIdx, int* d_colIdx) {
    std::vector<int> rowIdx(N + 1);
    std::vector<int> colIdx(nz);
    std::vector<float> val(nz);

    cudaMemcpy(rowIdx.data(), d_rowIdx, sizeof(int) * (N + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(colIdx.data(), d_colIdx, sizeof(int) * nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(val.data(), d_val, sizeof(float) * nz, cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> full(N, std::vector<float>(N, 0));

    // csr to full
    for (int i = 0; i < N; i++) {
        for (int j = rowIdx[i]; j < rowIdx[i + 1]; j++) {
            full[i][colIdx[j]] = val[j];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << full[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

CGSolver::CGSolver(int N) :N(N)
{
    if (cubHandle == NULL)
    {
        error_check(cublasCreate(&cubHandle));
    }


    // create cuSPARSE cusHandle
    if (cusHandle == NULL)
    {
        error_check(cusparseCreate(&cusHandle));
    }

    // create descriptor for matrix A
    error_check(cusparseCreateMatDescr(&descrA));

    // initialize properties of matrix A
    error_check(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    error_check(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    error_check(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
    error_check(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create descriptor for matrix L
    error_check(cusparseCreateMatDescr(&descrL));

    // initialize properties of matrix L
    error_check(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
    error_check(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
    error_check(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
    error_check(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT));
    error_check(cudaMalloc(&d_x, N * sizeof(float)));
    error_check(cudaMalloc(&d_y, N * sizeof(float)));
    error_check(cudaMalloc(&d_z, N * sizeof(float)));
    error_check(cudaMalloc(&d_r, N * sizeof(float)));
    error_check(cudaMalloc(&d_rt, N * sizeof(float)));
    error_check(cudaMalloc(&d_xt, N * sizeof(float)));
    error_check(cudaMalloc(&d_q, N * sizeof(float)));
    error_check(cudaMalloc(&d_p, N * sizeof(float)));
    error_check(cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int)));
    error_check(cudaMemset(d_x, 0, N * sizeof(float)));
    error_check(cudaMemset(d_y, 0, N * sizeof(float)));
    error_check(cudaMemset(d_z, 0, N * sizeof(float)));
    error_check(cudaMemset(d_r, 0, N * sizeof(float)));
    error_check(cudaMemset(d_rt, 0, N * sizeof(float)));
    error_check(cudaMemset(d_xt, 0, N * sizeof(float)));
    error_check(cudaMemset(d_q, 0, N * sizeof(float)));
    error_check(cudaMemset(d_p, 0, N * sizeof(float)));
}

CGSolver::~CGSolver()
{
    error_check(cudaFree(&d_x));
    error_check(cudaFree(&d_y));
    error_check(cudaFree(&d_z));
    error_check(cudaFree(&d_r));
    error_check(cudaFree(&d_rt));
    error_check(cudaFree(&d_xt));
    error_check(cudaFree(&d_q));
    error_check(cudaFree(&d_p));
}

void sort_coo(int N, int nz, float* d_A, int* d_rowIdx, int* d_colIdx) {
    // 将裸指针转换为 Thrust 的 device_ptr
    thrust::device_ptr<int> d_rowIdx_ptr(d_rowIdx);
    thrust::device_ptr<int> d_colIdx_ptr(d_colIdx);
    thrust::device_ptr<float> d_A_ptr(d_A);

    // 创建 zip_iterator
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_rowIdx_ptr, d_colIdx_ptr, d_A_ptr));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_rowIdx_ptr + nz, d_colIdx_ptr + nz, d_A_ptr + nz));

    // 按 rowIdx 排序，按 colIdx 次序排序
    thrust::sort(begin, end, thrust::less<thrust::tuple<int, int, float>>());
}

void CGSolver::Solve(int N, float* d_b, float* d_x, float* d_A, int nz, int* d_rowIdx, int* d_colIdx, float* d_guess)
{
    sort_coo(N, nz, d_A, d_rowIdx, d_colIdx);
    printSparseCOOToFull(N, nz, d_A, d_rowIdx, d_colIdx);
    cusparseXcoo2csr(cusHandle, d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    inspectGLM(d_rowPtrA, N + 1);
    printSparseCSRToFull(N, nz, d_A, d_rowPtrA, d_colIdx);
    error_check(cusparseCreateCsr(&spMatDescrA, N, N, nz, d_rowPtrA, d_colIdx, d_A, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnVecDescr_t dvec_p;
    error_check(cusparseCreateDnVec(&dvec_p, N, d_p, CUDA_R_32F));

    cusparseDnVecDescr_t dvec_q;
    error_check(cusparseCreateDnVec(&dvec_q, N, d_q, CUDA_R_32F));
    error_check(cusparseCreateDnVec(&dvec_x, N, d_x, CUDA_R_32F));
    error_check(cusparseCreateDnVec(&dvec_b, N, d_b, CUDA_R_32F));

    // Incomplete Cholesky factorization
    error_check(cudaMalloc(&d_ic, nz * sizeof(float)));
    error_check(cudaMemcpy(d_ic, d_A, nz * sizeof(float), cudaMemcpyDeviceToDevice));

    error_check(cusparseCreateCsric02Info(&ic02info));

    int ic02BufferSizeInBytes = 0;
    error_check(cusparseScsric02_bufferSize(cusHandle, N, nz, descrA, d_ic, d_rowPtrA, d_colIdx, ic02info, &ic02BufferSizeInBytes));

    void* ic02Buffer = nullptr;
    error_check(cudaMalloc(&ic02Buffer, ic02BufferSizeInBytes));
    error_check(cusparseScsric02_analysis(cusHandle, N, nz, descrA, d_ic, d_rowPtrA, d_colIdx, ic02info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, ic02Buffer));

    error_check(cusparseScsric02(cusHandle, N, nz, descrA, d_ic, d_rowPtrA, d_colIdx, ic02info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, ic02Buffer));
    error_check(cusparseCreateCsr(&spMatDescrL, N, N, nz, d_rowPtrA, d_colIdx, d_ic, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    printSparseCSRToFull(N, nz, d_ic, d_rowPtrA, d_colIdx);
    // Prepare 
    error_check(cusparseSpSV_createDescr(&spsvDescrL));
    error_check(cusparseSpSV_createDescr(&spsvDescrU));

    size_t tmpBufferSize = 0;
    size_t bufferSize = 0;
    error_check(cusparseSpSV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spMatDescrL, dvec_x, dvec_b, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &tmpBufferSize));
    error_check(cusparseSpSV_bufferSize(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, spMatDescrL, dvec_x, dvec_b, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSize));

    if (tmpBufferSize > bufferSize)
        bufferSize = tmpBufferSize;

    error_check(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spMatDescrA, dvec_p, &zero, dvec_q, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &tmpBufferSize));
    if (tmpBufferSize > bufferSize)
        bufferSize = tmpBufferSize;

    error_check(cudaMalloc(&d_buf, bufferSize));
    error_check(cudaMalloc(&d_buf1, bufferSize));

    error_check(cusparseSpSV_analysis(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spMatDescrL, dvec_x, dvec_b, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_buf));
    error_check(cusparseSpSV_analysis(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, spMatDescrL, dvec_x, dvec_b, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_buf1));

    // x = 0
    // r0 = b  (since x == 0, b - A*x = b)
    error_check(cudaMemcpy(d_r, d_b, N * sizeof(float), cudaMemcpyDeviceToDevice));
    inspectGLM(d_b, N);

    error_check(cusparseCreateDnVec(&dvec_r, N, d_r, CUDA_R_32F));
    error_check(cusparseCreateDnVec(&dvec_y, N, d_y, CUDA_R_32F));
    error_check(cusparseCreateDnVec(&dvec_z, N, d_z, CUDA_R_32F));

    if (d_guess != nullptr)
    {
        // x = guess
        error_check(cudaMemcpy(d_x, d_guess, N * sizeof(float), cudaMemcpyDeviceToDevice));
        // r0 = b - A*x
        //     q = A*x
        //     r0 = -q + b
        error_check(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spMatDescrA,
            dvec_x, &zero, dvec_q, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, d_buf));
        float n_one = -1;
        error_check(cublasSaxpy(cubHandle, N, &n_one, d_q, 1, d_r, 1));
    }

    for (k = 0; k < max_iter; ++k)
    {
        // if ||rk|| < tolerance
        error_check(cublasSnrm2(cubHandle, N, d_r, 1, &rTr));
        //std::cout << "Iteration " << k << ": " << rTr << std::endl;
        if (rTr < tolerance)
        {
            break;
        }
        // Solve L*y = rk
        error_check(cusparseSpSV_solve(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            spMatDescrL, dvec_r, dvec_y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));
        inspectGLM(d_y, N);

        // Solve L^T*zk = y
        error_check(cusparseSpSV_solve(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one,
            spMatDescrL, dvec_y, dvec_z, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));
        inspectGLM(d_z, N);

        // rho_t = r{k-1} * z{k-1}
        rho_t = rho;
        // rho = rk * zk
        error_check(cublasSdot(cubHandle, N, d_r, 1, d_z, 1, &rho));

        if (k == 0)
        {
            // pk = zk
            error_check(cublasScopy(cubHandle, N, d_z, 1, d_p, 1));
        }
        else
        {
            // beta = (rk*zk) / (r{k-1}*z{k-1})
            beta = rho / rho_t;
            // pk = zk + beta*p{k-1}
            error_check(cublasSscal(cubHandle, N, &beta, d_p, 1));
            error_check(cublasSaxpy(cubHandle, N, &one, d_z, 1, d_p, 1));
        }

        // q = A*pk
        error_check(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spMatDescrA,
            dvec_p, &zero, dvec_q, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, d_buf));

        // alpha = (rk*zk) / (pk*q)
        error_check(cublasSdot(cubHandle, N, d_p, 1, d_q, 1, &pTq));
        alpha = rho / pTq;

        // x{k+1} = xk + alpha*pk
        error_check(cublasSaxpy(cubHandle, N, &alpha, d_p, 1, d_x, 1));

        // r{k+1} = rk - alpha*q 
        float n_alpha = -alpha;
        error_check(cublasSaxpy(cubHandle, N, &n_alpha, d_q, 1, d_r, 1));
    }

    error_check(cusparseDestroySpMat(spMatDescrA));
    error_check(cusparseDestroySpMat(spMatDescrL));
    error_check(cusparseDestroyDnVec(dvec_p));
    error_check(cusparseDestroyDnVec(dvec_q));
    error_check(cusparseDestroyDnVec(dvec_x));
    error_check(cusparseDestroyCsric02Info(ic02info));
    error_check(cusparseSpSV_destroyDescr(spsvDescrL));
    error_check(cusparseSpSV_destroyDescr(spsvDescrU));
}
