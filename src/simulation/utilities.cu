#include <utilities.cuh>
#include <collision/aabb.h>
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

template void inspectSparseMatrix(float* dev_val, int* dev_rowIdx, int* dev_colIdx, int nnz, int size);
template void inspectSparseMatrix(double* dev_val, int* dev_rowIdx, int* dev_colIdx, int nnz, int size);

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
template __global__ void TransformVertices(glm::tvec3<float>* X, glm::mat4 transform, int numVerts);
template __global__ void TransformVertices(glm::tvec3<double>* X, glm::mat4 transform, int numVerts);

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

template __global__ void PopulatePos(glm::vec3* vertices, glm::tvec3<float>* X, indexType* Tet, int numTets);
template __global__ void PopulatePos(glm::vec3* vertices, glm::tvec3<double>* X, indexType* Tet, int numTets);

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

template __global__ void PopulateTriPos(glm::vec3* vertices, glm::tvec3<float>* X, indexType* Tet, int numTris);
template __global__ void PopulateTriPos(glm::vec3* vertices, glm::tvec3<double>* X, indexType* Tet, int numTris);


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
    std::vector<Query> hstQueries(size);
    cudaMemcpy(hstQueries.data(), dev_query, sizeof(Query) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(hstQueries.data(), size);
}

void inspectSphere(const Sphere* dev_spheres, int size)
{
    std::vector<Sphere> hstSphere(size);
    cudaMemcpy(hstSphere.data(), dev_spheres, sizeof(Sphere) * size, cudaMemcpyDeviceToHost);
    utilityCore::inspectHost(hstSphere.data(), size);
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