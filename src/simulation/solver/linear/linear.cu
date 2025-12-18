#include <linear/linear.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

template<typename T>
void sort_coo(int N, int& nz, T* d_A, int* d_rowIdx, int* d_colIdx, T*& new_A, int*& new_rowIdx, int*& new_colIdx, int& capacity) {
    if (nz > capacity) {
        if (new_rowIdx) cudaFree(new_rowIdx);
        if (new_colIdx) cudaFree(new_colIdx);
        if (new_A)      cudaFree(new_A);
        cudaMalloc(&new_rowIdx, nz * sizeof(int));
        cudaMalloc(&new_colIdx, nz * sizeof(int));
        cudaMalloc(&new_A, nz * sizeof(T));
        capacity = nz;
    }

    thrust::device_ptr<int> d_row(d_rowIdx);
    thrust::device_ptr<int> d_col(d_colIdx);
    thrust::device_ptr<T> d_val(d_A);

    thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(d_row, d_col, d_val)),
        thrust::make_zip_iterator(thrust::make_tuple(d_row + nz, d_col + nz, d_val + nz)));

    thrust::device_ptr<int> d_new_row(new_rowIdx);
    thrust::device_ptr<int> d_new_col(new_colIdx);
    thrust::device_ptr<T> d_new_val(new_A);

    int new_nz = thrust::reduce_by_key(
        thrust::make_zip_iterator(thrust::make_tuple(d_row, d_col)),
        thrust::make_zip_iterator(thrust::make_tuple(d_row, d_col)) + nz,
        d_val,
        thrust::make_zip_iterator(thrust::make_tuple(d_new_row, d_new_col)),
        d_new_val,
        thrust::equal_to< thrust::tuple<int, int> >(),
        thrust::plus<T>()
    ).first - thrust::make_zip_iterator(thrust::make_tuple(d_new_row, d_new_col));

    nz = new_nz;
}

template void sort_coo(int N, int& nz, float* d_A, int* d_rowIdx, int* d_colIdx, float*& new_A, int*& new_rowIdx, int*& new_colIdx, int& capacity);
template void sort_coo(int N, int& nz, double* d_A, int* d_rowIdx, int* d_colIdx, double*& new_A, int*& new_rowIdx, int*& new_colIdx, int& capacity);