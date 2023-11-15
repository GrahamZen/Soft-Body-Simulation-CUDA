#include <utilities.cuh>
#include <cuda.h>

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int number)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < number)
    {
        X[index] = glm::vec3(transform * glm::vec4(X[index], 1.f));
    }
}

// Add the current iteration's output to the overall image
__global__ void AddGravity(glm::vec3* Force, glm::vec3* V, float mass, int numVerts, bool jump)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        if (jump)
            V[index].y += 20.f;
    }
}

__device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const GLuint* Tet, int tet) {
    glm::mat3 ret(0.0f);
    ret[0] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4]];
    ret[1] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4]];
    ret[2] = X[Tet[tet * 4 + 3]] - X[Tet[tet * 4]];

    return ret;
}

__global__ void computeInvDm(glm::mat3* inv_Dm, int tet_number, const glm::vec3* X, const GLuint* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tet_number)
    {
        inv_Dm[index] = glm::inverse(Build_Edge_Matrix(X, Tet, index));
    }
}

__global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int tet_number, const GLuint* Tet) {
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tet < tet_number) {
        glm::vec3 sum = V[Tet[tet * 4]] + V[Tet[tet * 4 + 1]] + V[Tet[tet * 4 + 2]] + V[Tet[tet * 4 + 3]];

        for (int i = 0; i < 4; ++i) {
            int idx = Tet[tet * 4 + i];
            atomicAdd(&(V_sum[idx].x), sum.x - V[idx].x);
            atomicAdd(&(V_sum[idx].y), sum.y - V[idx].y);
            atomicAdd(&(V_sum[idx].z), sum.z - V[idx].z);
            atomicAdd(&(V_num[idx]), 3);
        }
    }
}

__global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int number, const GLuint* Tet, float blendAlpha) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < number) {
        V[i] = blendAlpha * V[i] + (1 - blendAlpha) * V_sum[i] / float(V_num[i]);
    }
}


__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int tet_number)
{
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tet < tet_number)
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

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* vertices, int number)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < number)
    {
        glm::vec3 v0v1 = vertices[index * 3 + 1] - vertices[index * 3 + 0];
        glm::vec3 v0v2 = vertices[index * 3 + 2] - vertices[index * 3 + 0];
        glm::vec3 nor = glm::cross(v0v1, v0v2);
        norms[index * 3 + 0] = glm::vec4(glm::normalize(nor), 0.f);
        norms[index * 3 + 1] = glm::vec4(glm::normalize(nor), 0.f);
        norms[index * 3 + 2] = glm::vec4(glm::normalize(nor), 0.f);
    }
}

__global__ void ComputeForces(glm::vec3* Force, const glm::vec3* X, const GLuint* Tet, int tet_number, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1) {
    int tet = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet >= tet_number) return;

    glm::mat3 F = Build_Edge_Matrix(X, Tet, tet) * inv_Dm[tet];
    glm::mat3 FtF = glm::transpose(F) * F;
    glm::mat3 G = (FtF - glm::mat3(1.0f)) * 0.5f;
    glm::mat3 S = G * (2.0f * stiffness_1) + glm::mat3(1.0f) * (stiffness_0 * trace(G));
    glm::mat3 forces = F * S * glm::transpose(inv_Dm[tet]) * (-1.0f / (6.0f * glm::determinant(inv_Dm[tet])));

    glm::vec3 force_0 = -glm::vec3(forces[0] + forces[1] + forces[2]);
    glm::vec3 force_1 = glm::vec3(forces[0]);
    glm::vec3 force_2 = glm::vec3(forces[1]);
    glm::vec3 force_3 = glm::vec3(forces[2]);

    atomicAdd(&(Force[Tet[tet * 4 + 0]].x), force_0.x);
    atomicAdd(&(Force[Tet[tet * 4 + 0]].y), force_0.y);
    atomicAdd(&(Force[Tet[tet * 4 + 0]].z), force_0.z);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].x), force_1.x);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].y), force_1.y);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].z), force_1.z);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].x), force_2.x);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].y), force_2.y);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].z), force_2.z);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].x), force_3.x);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].y), force_3.y);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].z), force_3.z);
}

__global__ void UpdateParticles(glm::vec3* X, glm::vec3* V, const glm::vec3* Force,
    int number, float mass, float dt, float damp,
    glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number) return;

    V[i] += Force[i] / mass * dt;
    V[i] *= damp;
    X[i] += V[i] * dt;

    float signedDis = glm::dot(X[i] - floorPos, floorUp);
    if (signedDis < 0 && glm::dot(V[i], floorUp) < 0) {
        X[i] -= signedDis * floorUp;
        glm::vec3 vN = glm::dot(V[i], floorUp) * floorUp;
        glm::vec3 vT = V[i] - vN;
        float mag_vT = glm::length(vT);
        float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, 0.0f);
        V[i] = -muN * vN + a * vT;
    }
}

__global__ void HandleFloorCollision(glm::vec3* X, glm::vec3* V,
    int number, glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number) return;

    float signedDis = glm::dot(X[i] - floorPos, floorUp);
    if (signedDis < 0 && glm::dot(V[i], floorUp) < 0) {
        X[i] -= signedDis * floorUp;
        glm::vec3 vN = glm::dot(V[i], floorUp) * floorUp;
        glm::vec3 vT = V[i] - vN;
        float mag_vT = glm::length(vT);
        float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, 0.0f);
        V[i] = -muN * vN + a * vT;
    }
}