#include <thrust/sort.h>
#include <thrust/device_ptr.h>

template<typename T>
void sort_coo(int N, int nz, T* d_A, int* d_rowIdx, int* d_colIdx) {
    thrust::device_ptr<int> d_rowIdx_ptr(d_rowIdx);
    thrust::device_ptr<int> d_colIdx_ptr(d_colIdx);
    thrust::device_ptr<T> d_A_ptr(d_A);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_rowIdx_ptr, d_colIdx_ptr, d_A_ptr));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_rowIdx_ptr + nz, d_colIdx_ptr + nz, d_A_ptr + nz));

    thrust::sort(begin, end, thrust::less<thrust::tuple<int, int, T>>());
}
