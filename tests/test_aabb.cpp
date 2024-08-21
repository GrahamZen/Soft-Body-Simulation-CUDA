#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/cg.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>

void readSparseMatrix(const std::string& filename, std::vector<int>& rowIdx, std::vector<int>& colIdx, std::vector<float>& val, int& N) {
    //format: "   (1,1)       1.0000", N lines
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }
    file >> N;
    char c;
    int row, col;
    float v;
    while (file >> c >> row >> c >> col >> c >> v) {
        rowIdx.push_back(row - 1);
        colIdx.push_back(col - 1);
        val.push_back(v);
    }
    file.close();
}

void readDenseVec(const std::string& filename, std::vector<float>& b) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }
    float v;
    while (file >> v) {
        b.push_back(v);
    }
    file.close();
}
void printSparseCOOToFull(int N, int nz, float* d_val, int* d_rowIdx, int* d_colIdx);
void printSparseCSRToFull(int N, int nz, float* d_val, int* d_rowIdx, int* d_colIdx);


TEST_CASE("SPMV Test", "[CG]") {
    //// A = [1, 0, 0, 0
    ////      0, 4, 0, 0
    ////      5, 0, 6, 0
    ////      0, 8, 0, 9]

    //// b = [1
    ////      8
    ////      23
    ////      52]

    //// Host problem definition
    //const int A_num_rows = 4;
    //const int A_num_cols = 4;
    //const int A_nnz = 9;
    //int       hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    //int       hA_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    //float     hA_values[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                              6.0f, 7.0f, 8.0f, 9.0f };
    //float     hX[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    //float     hY[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    //float     hY_result[] = { 19.0f, 8.0f, 51.0f, 52.0f };
    //float alpha = 1.0f;
    ////--------------------------------------------------------------------------
    //// Device memory management
    //int* dA_csrOffsets, * dA_columns, *dA_rows;
    //float* dA_values, * dX, * dY;
    //cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    //cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int));
    //cudaMalloc((void**)&dA_rows, A_nnz * sizeof(int));
    //cudaMalloc((void**)&dA_values, A_nnz * sizeof(float));
    //cudaMalloc((void**)&dX, A_num_cols * sizeof(float));
    //cudaMalloc((void**)&dY, A_num_rows * sizeof(float));

    //cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice);
    ////--------------------------------------------------------------------------
    //// CUSPARSE APIs
    //cusparseHandle_t handle = NULL;
    //cusparseSpMatDescr_t matA;
    //cusparseDnVecDescr_t vecX, vecY;
    //void* dBuffer = NULL;
    //size_t bufferSize = 0;
    //cusparseSpSVDescr_t spsvDescr;
    //cusparseCreate(&handle);
    //// Create sparse matrix A in CSR format
    //cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_num_cols, dA_rows, CUSPARSE_INDEX_BASE_ZERO);
    //printSparseCOOToFull(A_num_rows, A_nnz, dA_values, dA_rows, dA_columns);
    //cusparseXcoo2csr(handle, dA_rows, A_nnz, A_num_rows, dA_csrOffsets, CUSPARSE_INDEX_BASE_ZERO);
    //printSparseCSRToFull(A_num_rows, A_nnz, dA_values, dA_csrOffsets, dA_columns);
    //cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
    //    dA_csrOffsets, dA_columns, dA_values,
    //    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    //// Create dense vector X
    //cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F);
    //// Create dense vector y
    //cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F);
    //// Create opaque data structure, that holds analysis data between calls.
    //cusparseSpSV_createDescr(&spsvDescr);
    //// Specify Lower|Upper fill mode.
    //cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    //cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
    //// Specify Unit|Non-Unit diagonal type.
    //cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    //cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
    //// allocate an external buffer for analysis
    //cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //    &alpha, matA, vecX, vecY, CUDA_R_32F,
    //    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
    //    &bufferSize);
    //cudaMalloc(&dBuffer, bufferSize);
    //cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //    &alpha, matA, vecX, vecY, CUDA_R_32F,
    //    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer);
    //// execute SpSV
    //cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //    &alpha, matA, vecX, vecY, CUDA_R_32F,
    //    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);

    //// destroy matrix/vector descriptors
    //cusparseDestroySpMat(matA);
    //cusparseDestroyDnVec(vecX);
    //cusparseDestroyDnVec(vecY);
    //cusparseSpSV_destroyDescr(spsvDescr);
    //cusparseDestroy(handle);
    ////--------------------------------------------------------------------------
    //// device result check
    //cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    //int correct = 1;
    //for (int i = 0; i < A_num_rows; i++)
    //{
    //    if (hY[i] != hY_result[i])
    //    {                // direct floating point comparison is not
    //        correct = 0; // reliable
    //        break;
    //    }
    //}
    //if (correct)
    //    printf("spsv_csr_example test PASSED\n");
    //else
    //    printf("spsv_csr_example test FAILED: wrong result\n");
    //for (size_t i = 0; i < A_num_rows; i++)
    //{
    //    printf("x[%d] = %f\n", i, hY[i]);
    //}
    ////--------------------------------------------------------------------------
    //// device memory deallocation
    //cudaFree(dBuffer);
    //cudaFree(dA_csrOffsets);
    //cudaFree(dA_rows);
    //cudaFree(dA_columns);
    //cudaFree(dA_values);
    //cudaFree(dX);
    //cudaFree(dY);
}
TEST_CASE("CG Test", "[CG]") {
    int N;

    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<float> val;
    std::vector<float> b;
    std::vector<float> ans;

    readSparseMatrix("c:/Users/93125/Desktop/A.txt", rowIdx, colIdx, val, N);
    readDenseVec("c:/Users/93125/Desktop/b.txt", b);
    readDenseVec("c:/Users/93125/Desktop/x.txt", ans);
    CGSolver cg(N);

    std::vector<float> x(N, 0);

    int* dev_ARowIdx = nullptr, * dev_AColIdx = nullptr;
    float* dev_Aval = nullptr, * dev_b = nullptr, * dev_x = nullptr;
    cudaMalloc(&dev_ARowIdx, rowIdx.size() * sizeof(int));
    cudaMalloc(&dev_AColIdx, colIdx.size() * sizeof(int));
    cudaMalloc(&dev_Aval, val.size() * sizeof(float));
    cudaMalloc(&dev_b, N * sizeof(float));
    cudaMalloc(&dev_x, N * sizeof(float));
    cudaMemset(dev_x, 0, N * sizeof(float));

    cudaMemcpy(dev_ARowIdx, rowIdx.data(), rowIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_AColIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Aval, val.data(), val.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cg.Solve(N, dev_b, dev_x, dev_Aval, rowIdx.size(), dev_ARowIdx, dev_AColIdx);

    cudaDeviceSynchronize();
    cudaMemcpy(x.data(), dev_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        REQUIRE(x[i] == Catch::Approx(ans[i]).margin(1e-3));
    }
}