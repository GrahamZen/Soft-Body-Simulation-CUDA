#pragma once

#include <utilities.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <vector>

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line);

#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)

class BVHNode;
class Query;
class Sphere;
class Plane;
class Cylinder;

template <typename T>
void inspectGLM(T* dev_ptr, int size, const char* str = nullptr) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(host_ptr.data(), size, str);
}

template <typename T>
void inspectSparseMatrix(T* dev_val, int* dev_rowIdx, int* dev_colIdx, int nnz, int size) {
    std::vector<T> host_val(nnz);
    std::vector<int> host_rowIdx(nnz);
    std::vector<int> host_colIdx(nnz);
    cudaMemcpy(host_val.data(), dev_val, sizeof(T) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rowIdx.data(), dev_rowIdx, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_colIdx.data(), dev_colIdx, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(host_val, host_rowIdx, host_colIdx, size);
}

void inspectMortonCodes(const int* dev_mortonCodes, int numTets);
void inspectBVHNode(const BVHNode* dev_BVHNodes, int numTets);
void inspectBVH(const AABB* dev_aabbs, int size);
void inspectQuerys(const Query* dev_query, int size);
void inspectSphere(const Sphere* dev_spheres, int size);

template <typename T1, typename T2>
bool compareDevVSHost(const T1* dev_ptr, const T2* host_ptr2, int size) {
    std::vector<T1> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T1) * size, cudaMemcpyDeviceToHost);
    return utilityCore::compareHostVSHost(host_ptr.data(), reinterpret_cast<T1*>(host_ptr2), size);
}

template <typename T1, typename T2>
bool compareDevVSDev(const T1* dev_ptr, const T2* dev_ptr2, int size) {
    std::vector<T1> host_ptr(size);
    std::vector<T2> host_ptr2(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T1) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ptr2.data(), dev_ptr2, sizeof(T2) * size, cudaMemcpyDeviceToHost);
    return utilityCore::compareHostVSHost(host_ptr.data(), reinterpret_cast<T1*>(host_ptr2.data()), size);
}

template<typename HighP>
__global__ void TransformVertices(glm::tvec3<HighP>* X, glm::mat4 transform, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        X[index] = glm::tvec3<HighP>(transform * glm::tvec4<HighP>(X[index], 1.f));
    }
}

template<typename HighP>
__global__ void PopulatePos(glm::vec3* vertices, glm::tvec3<HighP>* X, indexType* Tet, int numTets)
{
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tet < numTets)
    {
        vertices[tet * 12 + 0] = X[Tet[tet * 4 + 0]];
        vertices[tet * 12 + 1] = X[Tet[tet * 4 + 2]];
        vertices[tet * 12 + 2] = X[Tet[tet * 4 + 1]];
        vertices[tet * 12 + 3] = X[Tet[tet * 4 + 0]];
        vertices[tet * 12 + 4] = X[Tet[tet * 4 + 3]];
        vertices[tet * 12 + 5] = X[Tet[tet * 4 + 2]];
        vertices[tet * 12 + 6] = X[Tet[tet * 4 + 0]];
        vertices[tet * 12 + 7] = X[Tet[tet * 4 + 1]];
        vertices[tet * 12 + 8] = X[Tet[tet * 4 + 3]];
        vertices[tet * 12 + 9] = X[Tet[tet * 4 + 1]];
        vertices[tet * 12 + 10] = X[Tet[tet * 4 + 2]];
        vertices[tet * 12 + 11] = X[Tet[tet * 4 + 3]];
    }
}

template<typename HighP>
__global__ void PopulateTriPos(glm::vec3* vertices, glm::tvec3<HighP>* X, indexType* Tet, int numTris)
{
    int tri = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tri < numTris)
    {
        vertices[tri * 3 + 0] = X[Tet[tri * 3 + 0]];
        vertices[tri * 3 + 1] = X[Tet[tri * 3 + 2]];
        vertices[tri * 3 + 2] = X[Tet[tri * 3 + 1]];
    }
}

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int numVerts);
__global__ void populateBVHNodeAABBPos(BVHNode* nodes, glm::vec3* pos, int numNodes);