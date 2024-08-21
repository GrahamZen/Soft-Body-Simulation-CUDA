#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/cg.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>

void readSparseMatrix(const std::string& filename, std::vector<int>& rowIdx, std::vector<int>& colIdx, std::vector<double>& val, int& N) {
    //format: "   (1,1)       1.0000", N lines
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }
    file >> N;
    char c;
    int row, col;
    double v;
    while (file >> c >> row >> c >> col >> c >> v) {
        rowIdx.push_back(row);
        colIdx.push_back(col);
        val.push_back(v);
    }
    file.close();
}

void readDenseVec(const std::string& filename, std::vector<double>& b) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }
    double v;
    while (file >> v) {
        b.push_back(v);
    }
    file.close();
}

TEST_CASE("CG Test", "[CG]") {
    int N;

    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<double> val;
    std::vector<double> b;
    std::vector<double> ans;

    readSparseMatrix("c:/Users/93125/Desktop/A.txt", rowIdx, colIdx, val, N);
    readDenseVec("c:/Users/93125/Desktop/b.txt", b);
    readDenseVec("c:/Users/93125/Desktop/x.txt", ans);
    CGSolver cg(N);

    std::vector<double> x(N, 0);

    int* dev_ARowIdx = nullptr, * dev_AColIdx = nullptr;
    double* dev_Aval = nullptr, * dev_b = nullptr, * dev_x = nullptr;
    cudaMalloc(&dev_ARowIdx, rowIdx.size() * sizeof(int));
    cudaMalloc(&dev_AColIdx, colIdx.size() * sizeof(int));
    cudaMalloc(&dev_Aval, val.size() * sizeof(double));
    cudaMalloc(&dev_b, N * sizeof(double));
    cudaMalloc(&dev_x, N * sizeof(double));
    cudaMemset(dev_x, 0, N * sizeof(double));

    cudaMemcpy(dev_ARowIdx, rowIdx.data(), rowIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_AColIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Aval, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    cg.Solve(N, dev_b, dev_x, dev_Aval, rowIdx.size(), dev_ARowIdx, dev_AColIdx, dev_b);

    cudaDeviceSynchronize();
    cudaMemcpy(x.data(), dev_x, N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        REQUIRE(x[i] == Catch::Approx(ans[i]).margin(1e-3));
    }
    cudaMemset(dev_x, 0, N * sizeof(double));
    cg.Solve(N, dev_b, dev_x, dev_Aval, rowIdx.size(), dev_ARowIdx, dev_AColIdx, dev_b);
    cudaMemcpy(x.data(), dev_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        REQUIRE(x[i] == Catch::Approx(ans[i]).margin(1e-3));
    }
    cudaMemset(dev_x, 0, N * sizeof(double));
    cg.Solve(N, dev_b, dev_x, dev_Aval, rowIdx.size(), dev_ARowIdx, dev_AColIdx, dev_b);
    cudaMemcpy(x.data(), dev_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        REQUIRE(x[i] == Catch::Approx(ans[i]).margin(1e-3));
    }
}