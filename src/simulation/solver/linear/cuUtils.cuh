#pragma once

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

template <typename T>
cublasStatus_t cublasscal(cublasHandle_t handle, int n, T alpha, T* x, int incx) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSscal(handle, n, &alpha, x, incx);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cublasDscal(handle, n, &alpha, x, incx);
    }
    else if constexpr (std::is_same<T, cuComplex>::value) {
        return cublasCscal(handle, n, reinterpret_cast<cuComplex*>(&alpha), reinterpret_cast<cuComplex*>(x), incx);
    }
    else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZscal(handle, n, reinterpret_cast<cuDoubleComplex*>(&alpha), reinterpret_cast<cuDoubleComplex*>(x), incx);
    }
    else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

template <typename T>
cublasStatus_t cublascopy(cublasHandle_t handle, int n, T* x, int incx, T* y, int incy) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasScopy(handle, n, x, incx, y, incy);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cublasDcopy(handle, n, x, incx, y, incy);
    }
    else if constexpr (std::is_same<T, cuComplex>::value) {
        return cublasCcopy(handle, n, x, incx, y, incy);
    }
    else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZcopy(handle, n, x, incx, y, incy);
    }
    else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}


template <typename T>
cublasStatus_t cublasdot(cublasHandle_t handle, int n, T* x, int incx, T* y, int incy, T* result) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
    else if constexpr (std::is_same<T, cuComplex>::value) {
        return cublasCdotu(handle, n, x, incx, y, incy, result);
    }
    else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZdotu(handle, n, x, incx, y, incy, result);
    }
    else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}


template <typename T>
cublasStatus_t cublasnrm2(cublasHandle_t handle, int n, T* x, int incx, T* result) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSnrm2(handle, n, x, incx, result);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cublasDnrm2(handle, n, x, incx, result);
    }
    else if constexpr (std::is_same<T, cuComplex>::value) {
        return cublasScnrm2(handle, n, x, incx, result);
    }
    else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
        return cublasDznrm2(handle, n, x, incx, result);
    }
    else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}
template <typename T>
cublasStatus_t cublasAxpy(cublasHandle_t handle, int N, T alpha, T* d_x, int incx, T* d_y, int incy) {

    if constexpr (std::is_same<T, float>::value) {
        return cublasSaxpy(handle, N, &alpha, d_x, incx, d_y, incy);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cublasDaxpy(handle, N, &alpha, d_x, incx, d_y, incy);
    }
    else if constexpr (std::is_same<T, cuComplex>::value) {
        return cublasCaxpy(handle, N, &alpha, d_x, incx, d_y, incy);
    }
    else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZaxpy(handle, N, &alpha, d_x, incx, d_y, incy);
    }
    else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

template <typename T>
cusparseStatus_t cusparsecsric02_bufferSize(
    cusparseHandle_t         handle,
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    T* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    csric02Info_t            info,
    int* pBufferSizeInBytes
) {
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cusparseDcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
    }
    else {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
}
template <typename T>
cusparseStatus_t cusparsecsric02(
    cusparseHandle_t         handle,
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    T* csrSortedValA_valM,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    csric02Info_t            info,
    cusparseSolvePolicy_t    policy,
    void* pBuffer
) {
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cusparseDcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
    }
    else {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
}
template <typename T>
cusparseStatus_t cusparsecsric02_analysis(
    cusparseHandle_t         handle,
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    const T* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    csric02Info_t            info,
    cusparseSolvePolicy_t    policy,
    void* pBuffer) {
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cusparseDcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
    }
    else {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
}
template <typename T>
cusolverStatus_t cusolverSpcsrcholBufferInfo(
    cusolverSpHandle_t       handle,
    int                      n,
    int                      nnzA,
    const cusparseMatDescr_t descrA,
    const T* csrValA,
    const int* csrRowPtrA,
    const int* csrColIndA,
    csrcholInfo_t            info,
    size_t* internalDataInBytes,
    size_t* workspaceInBytes) {
    if constexpr (std::is_same<T, float>::value) {
        return cusolverSpScsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cusolverSpDcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes);
    }
    else {
        return CUSOLVER_STATUS_NOT_SUPPORTED;
    }
}

template <typename T>
cusolverStatus_t cusolverSpcsrcholSolve(
    cusolverSpHandle_t handle,
    int                n,
    const T* b,
    T* x,
    csrcholInfo_t      info,
    void* pBuffer) {
    if constexpr (std::is_same<T, float>::value) {
        return cusolverSpScsrcholSolve(handle, n, b, x, info, pBuffer);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cusolverSpDcsrcholSolve(handle, n, b, x, info, pBuffer);
    }
    else {
        return CUSOLVER_STATUS_NOT_SUPPORTED;
    }
}

template <typename T>
cusolverStatus_t  cusolverSpcsrcholFactor(
    cusolverSpHandle_t       handle,
    int                      n,
    int                      nnzA,
    const cusparseMatDescr_t descrA,
    const T* csrValA,
    const int* csrRowPtrA,
    const int* csrColIndA,
    csrcholInfo_t            info,
    void* pBuffer) {
    if constexpr (std::is_same<T, float>::value) {
        return cusolverSpScsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
    }
    else if constexpr (std::is_same<T, double>::value) {
        return cusolverSpDcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
    }
    else {
        return CUSOLVER_STATUS_NOT_SUPPORTED;
    }
}
