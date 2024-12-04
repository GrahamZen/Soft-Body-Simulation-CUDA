#include <linear/cg.h>
#include <linear/cuUtils.cuh>
#include <iostream>

template<typename T>
CGSolver<T>::CGSolver(int N, int max_iter, T tolerance) : N(N), max_iter(max_iter), tolerance(tolerance)
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

    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_z, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_q, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_p, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_rowPtrA, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_y, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_z, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_r, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_q, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_p, 0, N * sizeof(T)));

    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_p, N, d_p, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_q, N, d_q, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_y, N, d_y, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_z, N, d_z, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_r, N, d_r, dType));
}

template<typename T>
CGSolver<T>::~CGSolver()
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
template<typename T>
void CGSolver<T>::Solve(int N, T* d_b, T* d_x, T* A, int nz, int* rowIdx, int* colIdx, T* d_guess)
{
    assert(d_b != nullptr);
    assert(d_x != nullptr);
    assert(A != nullptr);
    assert(rowIdx != nullptr);
    assert(colIdx != nullptr);
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));

    //==============================================================================
    // Sort the COO matrix by row index and convert it to CSR format
    sort_coo(N, nz, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    cusparseXcoo2csr(cusHandle, d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    CHECK_CUSPARSE(cusparseCreateCsr(&d_matA, N, N, nz, d_rowPtrA, d_colIdx, d_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dType));

    //==============================================================================
    // Create dense vectors for p, q, x, b, y, z, r
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_x, N, d_x, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_b, N, d_b, dType));
    // x = 0, r0 = b  (since x == 0, b - A*x = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(T), cudaMemcpyDeviceToDevice));

    //==============================================================================
    // L = ichol(A), L is a lower triangular matrix
    if (nz > old_nnz) {
        if (d_ic != nullptr)
            CHECK_CUDA(cudaFree(d_ic));
        CHECK_CUDA(cudaMalloc((void**)&d_ic, nz * sizeof(T)));
        std::cout << "d_ic malloc." << std::endl;
        old_nnz = nz;
    }

    CHECK_CUDA(cudaMemcpy(d_ic, d_A, nz * sizeof(T), cudaMemcpyDeviceToDevice));

    int ic02BufferSizeInBytes = 0;
    CHECK_CUSPARSE(cusparsecsric02_bufferSize(cusHandle, N, nz, descrA, d_ic,
        d_rowPtrA, d_colIdx, ic02info, &ic02BufferSizeInBytes));

    if (ic02BufferSizeInBytes > old_ic02BufferSizeInBytes)
    {
        if (ic02Buffer != nullptr)
            CHECK_CUDA(cudaFree(ic02Buffer));
        CHECK_CUDA(cudaMalloc((void**)&ic02Buffer, ic02BufferSizeInBytes));
        std::cout << "ic02Buffer malloc." << std::endl;
        old_ic02BufferSizeInBytes = ic02BufferSizeInBytes;
    }
    CHECK_CUSPARSE(cusparsecsric02_analysis(cusHandle, N, nz, descrA, d_ic,
        d_rowPtrA, d_colIdx, ic02info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, ic02Buffer));

    CHECK_CUSPARSE(cusparsecsric02(cusHandle, N, nz, descrA, d_ic,
        d_rowPtrA, d_colIdx, ic02info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, ic02Buffer));
    CHECK_CUSPARSE(cusparseCreateCsr(&d_matL, N, N, nz, d_rowPtrA, d_colIdx, d_ic,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dType));

    //============================================================================== 
    // Prepare workspace for solving L*y = b and L^T*z = y
    size_t bufferSizeL = 0;
    size_t bufferSizeU = 0;
    size_t tmpBufferSize = 0;

    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, dType, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, dType, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
        dvec_p, &zero, dvec_q, dType, CUSPARSE_SPMV_CSR_ALG1, &tmpBufferSize));
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
        dvec_x, dvec_b, dType, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufL));
    CHECK_CUSPARSE(cusparseSpSV_analysis(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, d_matL,
        dvec_x, dvec_b, dType, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufU));

    //==============================================================================
    // Set initial guess
    if (d_guess != nullptr)
    {
        // x = guess
        CHECK_CUDA(cudaMemcpy(d_x, d_guess, N * sizeof(T), cudaMemcpyDeviceToDevice));
        // r0 = b - A*x
        // q = A*x
        // r0 = -q + b
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_x, &zero, dvec_q, dType, CUSPARSE_SPMV_CSR_ALG1, d_bufL));
        CHECK_CUBLAS(cublasAxpy(cubHandle, N, (T)-1, d_q, 1, d_r, 1));
    }

    //==============================================================================
    // PCG solver Begin
    for (k = 0; k < max_iter; ++k)
    {
        // if ||rk|| < tolerance
        CHECK_CUBLAS(cublasnrm2(cubHandle, N, d_r, 1, &rTr));
        if (rTr < tolerance)
        {
            break;
        }
        // Solve L*y = rk
        CHECK_CUSPARSE(cusparseSpSV_solve(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            d_matL, dvec_r, dvec_y, dType, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        // Solve L^T*zk = y
        CHECK_CUSPARSE(cusparseSpSV_solve(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, &one,
            d_matL, dvec_y, dvec_z, dType, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

        // rho_t = r{k-1} * z{k-1}
        rho_t = rho;
        // rho = rk * zk
        CHECK_CUBLAS(cublasdot(cubHandle, N, d_r, 1, d_z, 1, &rho));

        if (k == 0)
        {
            // pk = zk
            CHECK_CUBLAS(cublascopy(cubHandle, N, d_z, 1, d_p, 1));
        }
        else
        {
            // beta = (rk*zk) / (r{k-1}*z{k-1})
            beta = rho / rho_t;
            // pk = zk + beta*p{k-1}
            CHECK_CUBLAS(cublasscal(cubHandle, N, beta, d_p, 1));
            CHECK_CUBLAS(cublasAxpy(cubHandle, N, (T)1, d_z, 1, d_p, 1));
        }

        // q = A*pk
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_p, &zero, dvec_q, dType, CUSPARSE_SPMV_CSR_ALG1, d_bufL));

        // alpha = (rk*zk) / (pk*q)
        CHECK_CUBLAS(cublasdot(cubHandle, N, d_p, 1, d_q, 1, &pTq));
        alpha = rho / pTq;

        // x{k+1} = xk + alpha*pk
        CHECK_CUBLAS(cublasAxpy(cubHandle, N, alpha, d_p, 1, d_x, 1));

        // r{k+1} = rk - alpha*q 
        CHECK_CUBLAS(cublasAxpy(cubHandle, N, -alpha, d_q, 1, d_r, 1));
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(d_matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(d_matL));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_b));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_x));
}

template class CGSolver<float>;
template class CGSolver<double>;