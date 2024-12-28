#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/jacobi.h>
#include <vector>
#include <matrix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <chrono>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

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
    std::vector<Eigen::Triplet<double>> triplets;

    for (int i = 0; i < nonZeroEntries; ++i) {
        int row = dist(gen);
        int col = dist(gen);

        // Ensure it's in the upper triangle to later enforce symmetry
        if (row > col) std::swap(row, col);

        double value = valueDist(gen);
        triplets.push_back(Eigen::Triplet<double>(row, col, value));
        triplets.push_back(Eigen::Triplet<double>(col, row, value));
    }

    // Add N to the diagonal to make the matrix positive-definite
    for (int i = 0; i < N; ++i) {
        // denseA(i, i) += N;
        triplets.push_back(Eigen::Triplet<double>(i, i, (double)N));
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    for (int k = 0; k < triplets.size(); ++k) {
        rowIdx.push_back(triplets[k].row());
        colIdx.push_back(triplets[k].col());
        val.push_back(triplets[k].value());
    }
}

TEST_CASE("JACOBI Test", "[JACOBI][.][SKIP]") {
    int N = 75000;
    int nz = 100000;
    int num_test = 1;

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
    JacobiSolver<double> jacobi(N, 100);
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
        jacobi.Solve(N, dev_b, dev_x, dev_Aval, rowIdx.size(), dev_ARowIdx, dev_AColIdx);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        gpu_time += milliseconds;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaMemcpy(x.data(), dev_x, N * sizeof(double), cudaMemcpyDeviceToHost);

        Eigen::VectorXd eCUDAx = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
        Eigen::VectorXd e = ex - eCUDAx;
        REQUIRE(e.isZero(1e-3));
    }
    std::cout << "Averge GPU Execution time: " << gpu_time / num_test << " ms" << std::endl;
    std::cout << "Averge CPU Execution time: " << cpu_time / num_test << " ms" << std::endl;

    cudaFree(dev_ARowIdx);
    cudaFree(dev_AColIdx);
    cudaFree(dev_Aval);
    cudaFree(dev_b);
    cudaFree(dev_x);
}

TEST_CASE("vector 9", "[tensor]") {
    glm::dmat3 DmInv(-0.100000, -0.000000, 0.100000,
        -0.000000, -0.100000, 0.000000,
        0.000000, 0.000000, 0.100000);
    DmInv = glm::transpose(DmInv);
    glm::mat3 U(-0.6208, 0.7763, -0.1091, -0.2820, -0.3509, -0.8929, -0.7315, -0.5236, 0.4368);
    U = glm::transpose(U);
    glm::mat3 V(-0.6501, 0.3252, 0.6867, -0.6324, 0.2694, -0.7263, -0.4212, -0.9064, 0.0305);
    V = glm::transpose(V);
    glm::mat3 S(2.0818, 0, 0, 0, 0.5707, 0, 0, 0, 0.2592);
    glm::mat3 T0(0, -1, 0, 1, 0, 0, 0, 0, 0);
    glm::mat3 T1(0, 0, 0, 0, 0, 1, 0, -1, 0);
    glm::mat3 T2(0, 0, 1, 0, 0, 0, -1, 0, 0);
    T0 = 1 / (float)sqrt(2) * U * T0 * glm::transpose(V);
    T1 = 1 / (float)sqrt(2) * U * T1 * glm::transpose(V);
    T2 = 1 / (float)sqrt(2) * U * T2 * glm::transpose(V);
    Vector9<double> t0(T0);
    Vector9<double> t1(T1);
    Vector9<double> t2(T2);
    double s0 = S[0][0];
    double s1 = S[1][1];
    double s2 = S[2][2];
    Matrix9<double> H(2);
    H = H - (4 / (s0 + s1)) * (Matrix9<double>(t0, t0));
    H = H - (4 / (s1 + s2)) * (Matrix9<double>(t1, t1));
    H = H - (4 / (s0 + s2)) * (Matrix9<double>(t2, t2));
    Matrix9x12<double> PFPx = ComputePFPx(DmInv);
    Matrix12<double> Hessian = PFPx.transpose() * H * PFPx;
    printMatrix(H, "H");

    Matrix12<double> Hessian2;
    ComputeHessian(static_cast<double*>(&DmInv[0][0]), H, Hessian2);
    printMatrix(Hessian - Hessian2, "hessian");
}
