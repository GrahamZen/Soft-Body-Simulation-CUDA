#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/cg.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <chrono>

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
void generateb(int N, std::vector<double>& b) {
    b.clear();
    b.resize(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> bDist(1.0, 10.0);

    for (int i = 0; i < N; ++i) {
        b[i] = bDist(gen);
    }
}

void generateSPDMatrixCOO(int N, int nonZeroEntries, std::vector<int>& rowIdx, std::vector<int>& colIdx, std::vector<double>& val,
    Eigen::SparseMatrix<double>& A) {
    A.resize(N, N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, N - 1);
    std::uniform_real_distribution<> valueDist(1.0, 10.0);

    // Clear the vectors in case they have previous data
    rowIdx.clear();
    colIdx.clear();
    val.clear();
    // Ensure the matrix is symmetric and positive-definite
    Eigen::MatrixXd denseA = Eigen::MatrixXd::Zero(N, N);

    for (int i = 0; i < nonZeroEntries; ++i) {
        int row = dist(gen);
        int col = dist(gen);

        // Ensure it's in the upper triangle to later enforce symmetry
        if (row > col) std::swap(row, col);

        double value = valueDist(gen);
        denseA(row, col) += value; // Accumulate values to avoid too small values
        denseA(col, row) += value;
    }

    // Add N to the diagonal to make the matrix positive-definite
    for (int i = 0; i < N; ++i) {
        denseA(i, i) += N;
    }

    // Convert the dense matrix to a sparse matrix
    A = denseA.sparseView();

    // Extract the COO format
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            rowIdx.push_back(it.row());
            colIdx.push_back(it.col());
            val.push_back(it.value());
        }
    }
}

TEST_CASE("CG Test", "[CG]") {
    int N = 75000;
    int nz = 10000;
    int num_test = 100;

    double cpu_time = 0;
    double gpu_time = 0;


    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<double> val;
    std::vector<double> b;
    std::vector<double> x(N, 0);

    // readSparseMatrix("c:/Users/93125/Desktop/A.txt", rowIdx, colIdx, val, N);
    // readDenseVec("c:/Users/93125/Desktop/b.txt", b);
    Eigen::SparseMatrix<double> A;
    generateSPDMatrixCOO(N, nz, rowIdx, colIdx, val, A);

    // Solve Ax = b
    Eigen::VectorXd bVec;
    Eigen::VectorXd ex;

    // Choose a solver, here we use SparseLU as an example
    CGSolver cg(N, 100, 1e-3);
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;

    int* dev_ARowIdx = nullptr, * dev_AColIdx = nullptr;
    double* dev_Aval = nullptr, * dev_b = nullptr, * dev_x = nullptr;
    cudaMalloc(&dev_ARowIdx, rowIdx.size() * sizeof(int));
    cudaMalloc(&dev_AColIdx, colIdx.size() * sizeof(int));
    cudaMalloc(&dev_Aval, val.size() * sizeof(double));
    cudaMalloc(&dev_x, N * sizeof(double));
    cudaMalloc(&dev_b, N * sizeof(double));
    cudaMemset(dev_x, 0, N * sizeof(double));

    cudaMemcpy(dev_ARowIdx, rowIdx.data(), rowIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_AColIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Aval, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    for (size_t j = 0; j < num_test; j++)
    {
        generateb(N, b);
        bVec = Eigen::Map<Eigen::VectorXd>(b.data(), b.size());
        cudaMemcpy(dev_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        // record cpu time
        auto cpu_start = std::chrono::high_resolution_clock::now();
        solver.compute(A);
        ex = solver.solve(bVec);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - cpu_start;
        cpu_time += elapsed.count() * 1000;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cg.Solve(N, dev_b, dev_x, dev_Aval, rowIdx.size(), dev_ARowIdx, dev_AColIdx);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        gpu_time += milliseconds;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaMemcpy(x.data(), dev_x, N * sizeof(double), cudaMemcpyDeviceToHost);

        Eigen::VectorXd eCUDAx = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
        REQUIRE((A * eCUDAx - bVec).isZero(1e-6));
        std::cout << "GPU Execution time: " << milliseconds << " ms" << std::endl;
        std::cout << "CPU Execution time: " << elapsed.count() * 1000 << " ms" << std::endl;
    }
    std::cout << "Averge GPU Execution time: " << gpu_time / num_test << " ms" << std::endl;
    std::cout << "Averge CPU Execution time: " << cpu_time / num_test << " ms" << std::endl;
        
    cudaFree(dev_ARowIdx);
    cudaFree(dev_AColIdx);
    cudaFree(dev_Aval);
    cudaFree(dev_b);
    cudaFree(dev_x);
}