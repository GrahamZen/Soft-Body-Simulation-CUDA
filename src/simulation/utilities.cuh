#include <cuda.h>
#include <glm/glm.hpp>
#include <vector>
#include <GL/glew.h>

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line);

template <typename T>
void inspectGLM(T* dev_ptr, int size) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    inspectHost(host_ptr.data(), size);
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

__device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const GLuint* Tet, int tet);

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int number);
__global__ void AddGravity(glm::vec3* Force, glm::vec3* V, float mass, int numVerts, bool jump);
__global__ void computeInvDm(glm::mat3* inv_Dm, int tet_number, const glm::vec3* X, const GLuint* Tet);
__global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int tet_number, const GLuint* Tet);
__global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int number, const GLuint* Tet, float blendAlpha);
__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int tet_number);
__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int number);
__global__ void ComputeForces(glm::vec3* Force, const glm::vec3* X, const GLuint* Tet, int tet_number, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1);
__global__ void UpdateParticles(glm::vec3* X, glm::vec3* V, const glm::vec3* Force,
    int number, float mass, float dt, float damp,
    glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN);