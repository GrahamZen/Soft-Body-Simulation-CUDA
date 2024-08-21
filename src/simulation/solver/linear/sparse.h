#pragma once

struct SparseMatrix {
    // coo format
    int* rowIdx;
    int* colIdx;
    float* val;
    int N;

    int nnz;
};