#include <utilities.cuh>
#include <cuda.h>
#include <bvh.h>
#include <sphere.h>

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

void inspectMortonCodes(const int* dev_mortonCodes, int numTets) {
    std::vector<unsigned int> hstMorton(numTets);
    cudaMemcpy(hstMorton.data(), dev_mortonCodes, numTets * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    utilityCore::inspectHostMorton(hstMorton.data(), numTets);
}

void inspectBVHNode(const BVHNode* dev_BVHNodes, int numTets)
{
    std::vector<BVHNode> hstBVHNodes(2 * numTets - 1);
    cudaMemcpy(hstBVHNodes.data(), dev_BVHNodes, sizeof(BVHNode) * (2 * numTets - 1), cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(hstBVHNodes.data(), 2 * numTets - 1);
}

void inspectBVH(const AABB* dev_aabbs, int size)
{
    std::vector<AABB> hstAABB(size);
    cudaMemcpy(hstAABB.data(), dev_aabbs, sizeof(AABB) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(hstAABB.data(), size);
}

void inspectQuerys(const Query* dev_query, int size)
{
    std::vector<Query> hstAABB(size);
    cudaMemcpy(hstAABB.data(), dev_query, sizeof(Query) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(hstAABB.data(), size);
}

void inspectSphere(const Sphere* dev_spheres, int size)
{
    std::vector<Sphere> hstSphere(size);
    cudaMemcpy(hstSphere.data(), dev_spheres, sizeof(Sphere) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(hstSphere.data(), size);
}

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        X[index] = glm::vec3(transform * glm::vec4(X[index], 1.f));
    }
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        if (jump)
            V[index] += vel / mass;
    }
}

__device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const GLuint* Tet, int tet) {
    glm::mat3 ret(0.0f);
    ret[0] = X[Tet[tet * 4]] - X[Tet[tet * 4 + 3]];
    ret[1] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4 + 3]];
    ret[2] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4 + 3]];

    return ret;
}

__global__ void computeInvDmV0(float* V0, glm::mat3* inv_Dm, int numTets, const glm::vec3* X, const GLuint* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
        inv_Dm[index] = glm::inverse(Dm);
        V0[index] = glm::abs(glm::determinant(Dm)) / 6.0f;
    }
}

__global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numTets, const GLuint* Tet) {
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tet < numTets) {
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

__global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numVerts, const GLuint* Tet, float blendAlpha) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < numVerts) {
        V[i] = blendAlpha * V[i] + (1 - blendAlpha) * V_sum[i] / float(V_num[i]);
    }
}


__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int numTets)
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

__global__ void PopulateTriPos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int numTris)
{
    int tri = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tri < numTris)
    {
        vertices[tri * 3 + 0] = X[Tet[tri * 3 + 0]];
        vertices[tri * 3 + 1] = X[Tet[tri * 3 + 2]];
        vertices[tri * 3 + 2] = X[Tet[tri * 3 + 1]];
    }
}

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* vertices, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        glm::vec3 v0v1 = vertices[index * 3 + 1] - vertices[index * 3 + 0];
        glm::vec3 v0v2 = vertices[index * 3 + 2] - vertices[index * 3 + 0];
        glm::vec3 nor = glm::cross(v0v1, v0v2);
        norms[index * 3 + 0] = glm::vec4(glm::normalize(nor), 0.f);
        norms[index * 3 + 1] = glm::vec4(glm::normalize(nor), 0.f);
        norms[index * 3 + 2] = glm::vec4(glm::normalize(nor), 0.f);
    }
}

__global__ void ComputeForces(glm::vec3* Force, const glm::vec3* X, const GLuint* Tet, int numTets, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1) {
    int tet = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet >= numTets) return;

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

//simple example of gravity
__global__ void setExtForce(glm::vec3* ExtForce, glm::vec3 gravity, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        ExtForce[index] = gravity;
    }
}

__global__ void populateBVHNodeAABBPos(BVHNode* nodes, glm::vec3* pos, int numNodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numNodes) return;
    const AABB& aabb = nodes[idx].bbox;
    pos[idx * 8 + 0] = glm::vec3(aabb.min.x, aabb.min.y, aabb.max.z);
    pos[idx * 8 + 1] = glm::vec3(aabb.max.x, aabb.min.y, aabb.max.z);
    pos[idx * 8 + 2] = glm::vec3(aabb.max.x, aabb.max.y, aabb.max.z);
    pos[idx * 8 + 3] = glm::vec3(aabb.min.x, aabb.max.y, aabb.max.z);
    pos[idx * 8 + 4] = glm::vec3(aabb.min.x, aabb.min.y, aabb.min.z);
    pos[idx * 8 + 5] = glm::vec3(aabb.max.x, aabb.min.y, aabb.min.z);
    pos[idx * 8 + 6] = glm::vec3(aabb.max.x, aabb.max.y, aabb.min.z);
    pos[idx * 8 + 7] = glm::vec3(aabb.min.x, aabb.max.y, aabb.min.z);
}