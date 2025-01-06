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

template<typename Scalar>
class BVHNode;
class Query;
class Sphere;
class Plane;
class Cylinder;

template <typename T>
void inspectGLM(const T* dev_ptr, int size, const char* str = "") {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(host_ptr.data(), size, str);
}

template <typename T>
void inspectSparseMatrix(T* dev_val, int* dev_rowIdx, int* dev_colIdx, int begin, int nnz, int size);
void inspectMortonCodes(const int* dev_mortonCodes, int numTris);

template<typename Scalar>
void inspectBVHNode(const BVHNode<Scalar>* dev_BVHNodes, int numTris);
template<typename Scalar>
void inspectBVH(const AABB<Scalar>* dev_aabbs, int size);
void inspectQuerys(const Query* dev_query, int size);
void inspectSphere(const Sphere* dev_spheres, int size);

template <typename T1, typename T2>
bool compareDevVSHost(const T1* dev_ptr, const T2* host_ptr2, int size);
template <typename T1, typename T2>
bool compareDevVSDev(const T1* dev_ptr, const T2* dev_ptr2, int size);

template<typename Scalar>
__global__ void TransformVertices(glm::tvec3<Scalar>* X, glm::mat4 transform, int numVerts);

template<typename Scalar>
__global__ void PopulatePos(glm::vec3* vertices, glm::tvec3<Scalar>* X, indexType* Tet, int numTets);

template<typename Scalar>
__global__ void PopulateTriPos(glm::vec3* vertices, glm::tvec3<Scalar>* X, indexType* Tet, int numTris);

__host__ __device__ glm::vec4 getNormal(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2);

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int numVerts);
template<typename Scalar>
__global__ void populateBVHNodeAABBPos(BVHNode<Scalar>* nodes, glm::vec3* pos, int numNodes);