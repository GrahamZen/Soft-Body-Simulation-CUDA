#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include <vector>
#include <GL/glew.h>
#include <utilities.h>
#include <cuda_runtime.h>

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
void inspectGLM(T* dev_ptr, int size) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(host_ptr.data(), size);
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

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int numVerts);
__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, indexType* Tet, int numTets);
__global__ void PopulateTriPos(glm::vec3* vertices, glm::vec3* X, indexType* Tet, int numTris);
__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int numVerts);

__global__ void handleSphereCollision(glm::vec3* X, glm::vec3* V, int numVerts, Sphere* spheres, int numSpheres, float muT, float muN);
__global__ void handleFloorCollision(glm::vec3* X, glm::vec3* V, int numVerts, Plane* planes, int numPlanes, float muT, float muN);
__global__ void handleCylinderCollision(glm::vec3* X, glm::vec3* V, int numVerts, Cylinder* cylinders, int numCylinders, float muT, float muN);

__global__ void populateBVHNodeAABBPos(BVHNode* nodes, glm::vec3* pos, int numNodes);