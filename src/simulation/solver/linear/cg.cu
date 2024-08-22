#include <linear/cg.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
    }                                                                          \
}

CGSolver::CGSolver(int N, int max_iter, double tolerance) : N(N), max_iter(max_iter), tolerance(tolerance)
{
    CHECK_CUBLAS(cublasCreate(&cubHandle));

    CHECK_CUSPARSE(cusparseCreate(&cusHandle));
    // create descriptor for matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    // initialize properties of matrix A
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create descriptor for matrix L
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrL));

    // initialize properties of matrix L
    CHECK_CUSPARSE(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT));

    CHECK_CUSPARSE(cusparseCreateCsric02Info(&ic02info));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrU));

    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_z, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_q, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_rowPtrA, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_y, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_z, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_r, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_q, 0, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_p, 0, N * sizeof(double)));

    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_p, N, d_p, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_q, N, d_q, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_y, N, d_y, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_z, N, d_z, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_r, N, d_r, CUDA_R_64F));
}

CGSolver::~CGSolver()
{
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_rowPtrA));
    CHECK_CUDA(cudaFree(d_ic));
    CHECK_CUDA(cudaFree(d_bufL));
    CHECK_CUDA(cudaFree(d_bufU));

    CHECK_CUBLAS(cublasDestroy(cubHandle));
    CHECK_CUSPARSE(cusparseDestroy(cusHandle));

    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_r));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_p));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_q));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_y));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_z));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrL));
    CHECK_CUSPARSE(cusparseDestroyCsric02Info(ic02info));
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrL));
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrU));
}

void sort_coo(int N, int nz, double* d_A, int* d_rowIdx, int* d_colIdx) {
    thrust::device_ptr<int> d_rowIdx_ptr(d_rowIdx);
    thrust::device_ptr<int> d_colIdx_ptr(d_colIdx);
    thrust::device_ptr<double> d_A_ptr(d_A);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_rowIdx_ptr, d_colIdx_ptr, d_A_ptr));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_rowIdx_ptr + nz, d_colIdx_ptr + nz, d_A_ptr + nz));

    thrust::sort(begin, end, thrust::less<thrust::tuple<int, int, double>>());
}

void CGSolver::Solve(int N, double* d_b, double* d_x, double* d_A, int nz, int* d_rowIdx, int* d_colIdx, double* d_guess)
{
    assert(d_b != nullptr);
    assert(d_x != nullptr);
    assert(d_A != nullptr);
    assert(d_rowIdx != nullptr);
    assert(d_colIdx != nullptr);
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(double)));

    //==============================================================================
    // Sort the COO matrix by row index and convert it to CSR format
    sort_coo(N, nz, d_A, d_rowIdx, d_colIdx);
    cusparseXcoo2csr(cusHandle, d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    CHECK_CUSPARSE(cusparseCreateCsr(&d_matA, N, N, nz, d_rowPtrA, d_colIdx, d_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    //==============================================================================
    // Create dense vectors for p, q, x, b, y, z, r
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_x, N, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_b, N, d_b, CUDA_R_64F));
    // x = 0, r0 = b  (since x == 0, b - A*x = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice));

    //==============================================================================
    // L = ichol(A), L is a lower triangular matrix
    if (nz > old_nnz) {
        if (d_ic != nullptr)
            CHECK_CUDA(cudaFree(d_ic));
        CHECK_CUDA(cudaMalloc((void**)&d_ic, nz * sizeof(double)));
        std::cout << "d_ic malloc." << std::endl;
        old_nnz = nz;
    }

    CHECK_CUDA(cudaMemcpy(d_ic, d_A, nz * sizeof(double), cudaMemcpyDeviceToDevice));

    int ic02BufferSizeInBytes = 0;
    CHECK_CUSPARSE(cusparseDcsric02_bufferSize(cusHandle, N, nz, descrA, d_ic,
        d_rowPtrA, d_colIdx, ic02info, &ic02BufferSizeInBytes));

    if (ic02BufferSizeInBytes > old_ic02BufferSizeInBytes)
    {
        if (ic02Buffer != nullptr)
            CHECK_CUDA(cudaFree(ic02Buffer));
        CHECK_CUDA(cudaMalloc((void**)&ic02Buffer, ic02BufferSizeInBytes));
        std::cout << "ic02Buffer malloc." << std::endl;
        old_ic02BufferSizeInBytes = ic02BufferSizeInBytes;
    }
    CHECK_CUSPARSE(cusparseDcsric02_analysis(cusHandle, N, nz, descrA, d_ic,
        d_rowPtrA, d_colIdx, ic02info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, ic02Buffer));

    CHECK_CUSPARSE(cusparseDcsric02(cusHandle, N, nz, descrA, d_ic,
        d_rowPtrA, d_colIdx, ic02info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, ic02Buffer));
    CHECK_CUSPARSE(cusparseCreateCsr(&d_matL, N, N, nz, d_rowPtrA, d_colIdx, d_ic,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    //============================================================================== 
    // Prepare workspace for solving L*y = b and L^T*z = y
    size_t bufferSizeL = 0;
    size_t bufferSizeU = 0;
    size_t tmpBufferSize = 0;

    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
        dvec_p, &zero, dvec_q, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &tmpBufferSize));
    if (tmpBufferSize > bufferSizeL)
        bufferSizeL = tmpBufferSize;

    if (bufferSizeL > old_bufferSizeL)
    {
        if (d_bufL != nullptr)
            CHECK_CUDA(cudaFree(d_bufL));
        CHECK_CUDA(cudaMalloc((void**)&d_bufL, bufferSizeL));
        std::cout << "d_bufL malloc." << std::endl;
        old_bufferSizeL = bufferSizeL;
    }

    if (bufferSizeU > old_bufferSizeU)
    {
        if (d_bufU != nullptr)
            CHECK_CUDA(cudaFree(d_bufU));
        CHECK_CUDA(cudaMalloc((void**)&d_bufU, bufferSizeU));
        std::cout << "d_bufU malloc." << std::endl;
        old_bufferSizeU = bufferSizeU;
    }

    CHECK_CUSPARSE(cusparseSpSV_analysis(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufL));
    CHECK_CUSPARSE(cusparseSpSV_analysis(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufU));

    //==============================================================================
    // Set initial guess
    if (d_guess != nullptr)
    {
        // x = guess
        CHECK_CUDA(cudaMemcpy(d_x, d_guess, N * sizeof(double), cudaMemcpyDeviceToDevice));
        // r0 = b - A*x
        // q = A*x
        // r0 = -q + b
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_x, &zero, dvec_q, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, d_bufL));
        double n_one = -1;
        CHECK_CUBLAS(cublasDaxpy(cubHandle, N, &n_one, d_q, 1, d_r, 1));
    }

    //==============================================================================
    // PCG solver Begin
    for (k = 0; k < max_iter; ++k)
    {
        // if ||rk|| < tolerance
        CHECK_CUBLAS(cublasDnrm2(cubHandle, N, d_r, 1, &rTr));
        if (rTr < tolerance)
        {
            break;
        }
        // Solve L*y = rk
        CHECK_CUSPARSE(cusparseSpSV_solve(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            d_matL, dvec_r, dvec_y, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        // Solve L^T*zk = y
        CHECK_CUSPARSE(cusparseSpSV_solve(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one,
            d_matL, dvec_y, dvec_z, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

        // rho_t = r{k-1} * z{k-1}
        rho_t = rho;
        // rho = rk * zk
        CHECK_CUBLAS(cublasDdot(cubHandle, N, d_r, 1, d_z, 1, &rho));

        if (k == 0)
        {
            // pk = zk
            CHECK_CUBLAS(cublasDcopy(cubHandle, N, d_z, 1, d_p, 1));
        }
        else
        {
            // beta = (rk*zk) / (r{k-1}*z{k-1})
            beta = rho / rho_t;
            // pk = zk + beta*p{k-1}
            CHECK_CUBLAS(cublasDscal(cubHandle, N, &beta, d_p, 1));
            CHECK_CUBLAS(cublasDaxpy(cubHandle, N, &one, d_z, 1, d_p, 1));
        }

        // q = A*pk
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_p, &zero, dvec_q, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, d_bufL));

        // alpha = (rk*zk) / (pk*q)
        CHECK_CUBLAS(cublasDdot(cubHandle, N, d_p, 1, d_q, 1, &pTq));
        alpha = rho / pTq;

        // x{k+1} = xk + alpha*pk
        CHECK_CUBLAS(cublasDaxpy(cubHandle, N, &alpha, d_p, 1, d_x, 1));

        // r{k+1} = rk - alpha*q 
        double n_alpha = -alpha;
        CHECK_CUBLAS(cublasDaxpy(cubHandle, N, &n_alpha, d_q, 1, d_r, 1));
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(d_matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(d_matL));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_b));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_x));
}