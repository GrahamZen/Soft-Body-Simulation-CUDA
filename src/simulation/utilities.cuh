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

__inline__ __device__ float trace(const glm::mat3& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

__inline__ __device__ float trace2(const glm::mat3& a)
{
    return (float)((a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]));
}

__inline__ __device__ float trace4(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2]);
}

__inline__ __device__ float det2(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2]);
}

__inline__ __device__ void setRowColVal(int index, int* row, int* col, float* val, int r, int c, float v)
{
    row[index] = r;
    col[index] = c;
    val[index] = v;
}

__inline__ __device__ void wrapRowColVal(int index, int* idx, float* val, int r, int c, float v, int rowLen)
{
    idx[index] = r * rowLen + c;
    val[index] = v;
}


__device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const GLuint* Tet, int tet);
__device__ void svdGLM(const glm::mat3& A, glm::mat3& U, glm::mat3& S, glm::mat3& V);

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int numVerts);
__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);
__global__ void computeInvDmV0(float* V0, glm::mat3* inv_Dm, int numTets, const glm::vec3* X, const GLuint* Tet);
__global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numTets, const GLuint* Tet);
__global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numVerts, const GLuint* Tet, float blendAlpha);
__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int numTets);
__global__ void PopulateTriPos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int numTris);
__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int numVerts);
__global__ void ComputeForces(glm::vec3* Force, const glm::vec3* X, const GLuint* Tet, int numTets, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1);
__global__ void UpdateParticles(glm::vec3* X, glm::vec3* V, const glm::vec3* Force,
    int numVerts, float mass, float dt, float damp,
    glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN);

__global__ void handleSphereCollision(glm::vec3* X, glm::vec3* V, int numVerts, Sphere* spheres, int numSpheres, float muT, float muN);
__global__ void handleFloorCollision(glm::vec3* X, glm::vec3* V, int numVerts, Plane* planes, int numPlanes, float muT, float muN);
__global__ void handleCylinderCollision(glm::vec3* X, glm::vec3* V, int numVerts, Cylinder* cylinders, int numCylinders, float muT, float muN);

__global__ void computeLocal(const float* V0, const float wi, float* xProj, const glm::mat3* DmInv, const float* qn__1, const GLuint* tetIndex, int tetNumber);
__global__ void computeSn(float* sn, float dt, float dt2_m_1, glm::vec3* pos, glm::vec3* vel, const glm::vec3* force, float* b, float massDt_2, int numVerts);
__global__ void setMDt_2(int* outIdx, float* val, int startIndex, float massDt_2, int vertNumber);
__global__ void computeM_h2Sn(float* b, float* sn, float massDt_2, int vertNumber);
__global__ void addM_h2Sn(float* b, float* masses, int vertNumber);
__global__ void computeSiTSi(int* outIdx, float* val, float* V0, glm::mat3* DmInv, GLuint* tetIndex, float weight, int tetNumber, int vertNumber);
__global__ void updateVelPos(float* newPos, float dt_1, glm::vec3* pos, glm::vec3* vel, int numVerts);
__global__ void initAMatrix(int* idx, int* row, int* col, int rowLen, int totalNumber);
__global__ void setExtForce(glm::vec3* ExtForce, glm::vec3 gravity, int numVerts);

__global__ void populateBVHNodeAABBPos(BVHNode* nodes, glm::vec3* pos, int numNodes);