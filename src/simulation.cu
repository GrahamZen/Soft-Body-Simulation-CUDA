/**
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "simulation.h"

#include "deformable_mesh.h"
#include "solver.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
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
void inspect(T* dev_ptr, int size) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    std::cout << std::endl;
}

__device__ float trace(const glm::mat3& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

__device__ float trace2(const glm::mat3& a)
{
    return (float)((a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]));
}

__device__ float trace4(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2]);
}

__device__ float det2(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2]);
}

static GuiDataContainer* guiData = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Add the current iteration's output to the overall image
__global__ void AddGravity(glm::vec3* Force, glm::vec3* V, float mass, int numVerts, bool jump)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        if (jump)
            V[index].y += 0.2f;
        Force[index] = glm::vec3(0, -9.8f, 0) * mass;
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
        inv_Dm[index] = Build_Edge_Matrix(X, Tet, index);
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

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int number)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < number)
    {
        glm::vec3 v0v1 = X[index * 3 + 2] - X[index * 3 + 1];
        glm::vec3 v0v2 = X[index * 3 + 2] - X[index * 3 + 0];
        glm::vec3 nor = glm::cross(v0v1, v0v2);
        norms[index * 3 + 0] = glm::vec4(nor, 1.f);
        norms[index * 3 + 1] = glm::vec4(nor, 1.f);
        norms[index * 3 + 2] = glm::vec4(nor, 1.f);
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
    atomicAdd(&(Force[Tet[tet * 4 + 1]].x), force_0.x);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].y), force_0.y);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].z), force_0.z);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].x), force_0.x);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].y), force_0.y);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].z), force_0.z);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].x), force_0.x);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].y), force_0.y);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].z), force_0.z);
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


SimulationCUDAContext::SimulationCUDAContext()
{
}

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto softbody : softBodies) {
        delete softbody;
    }
}

void SimulationCUDAContext::Update()
{
    for (auto softbody : softBodies) {
        softbody->Update();
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->mapDevicePtr(&pos, &nor);
        dim3 numThreadsPerBlock(softbody->getTetNumber() / 32 + 1);
        PopulatePos << <numThreadsPerBlock, 32 >> > (pos, softbody->getX(), softbody->getTet(), softbody->getTetNumber());
        RecalculateNormals << <softbody->getNumber() / 32 + 1, 32 >> > (nor, softbody->getX(), softbody->getNumber());
        softbody->unMapDevicePtr();
    }
}

SoftBody::SoftBody(const char* nodeFileName, const char* eleFileName) :Mesh()
{
    std::vector<glm::vec3> vertices = loadNodeFile(nodeFileName);
    number = vertices.size();
    cudaMalloc((void**)&X, sizeof(glm::vec3) * number);
    cudaMemcpy(X, vertices.data(), sizeof(glm::vec3) * number, cudaMemcpyHostToDevice);


    std::vector<GLuint> idx = loadEleFile(eleFileName);
    tet_number = idx.size() / 4;
    cudaMalloc((void**)&Tet, sizeof(GLuint) * idx.size());
    cudaMemcpy(Tet, idx.data(), sizeof(GLuint) * idx.size(), cudaMemcpyHostToDevice);

    Mesh::tet_number = tet_number;

    cudaMalloc((void**)&Force, sizeof(glm::vec3) * number);
    cudaMemset(Force, 0, sizeof(glm::vec3) * number);
    cudaMalloc((void**)&V, sizeof(glm::vec3) * number);
    cudaMemset(V, 0, sizeof(glm::vec3) * number);
    cudaMalloc((void**)&inv_Dm, sizeof(glm::mat4) * tet_number);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * number);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    createTetrahedron();
    cudaMalloc((void**)&V_num, sizeof(int) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDm << < blocks, threadsPerBlock >> > (inv_Dm, tet_number, X, Tet);
}

SoftBody::~SoftBody()
{
    cudaFree(X);
    cudaFree(Tet);
    cudaFree(Force);
    cudaFree(V);
    cudaFree(inv_Dm);
    cudaFree(V_sum);
}

void SoftBody::mapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);

    cudaGraphicsMapResources(1, &cuda_bufNor_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufNorDevPtr, &size, cuda_bufNor_resource);
}

void SoftBody::unMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_bufNor_resource, 0);
}

void SoftBody::Laplacian_Smoothing(float blendAlpha)
{
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    LaplacianGatherKern << < blocks, threadsPerBlock >> > (V, V_sum, V_num, tet_number, Tet);
    LaplacianKern << < (number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, V_sum, V_num, number, Tet, blendAlpha);
}

std::vector<GLuint> SoftBody::loadEleFile(const std::string& EleFilename)
{
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> tet_number;

    std::vector<GLuint> Tet(tet_number * 4);

    int a, b, c, d, e;
    for (int tet = 0; tet < tet_number && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Tet[tet * 4 + 0] = b - 1;
        Tet[tet * 4 + 1] = c - 1;
        Tet[tet * 4 + 2] = d - 1;
        Tet[tet * 4 + 3] = e - 1;
    }

    file.close();
    return Tet;
}

std::vector<glm::vec3> SoftBody::loadNodeFile(const std::string& nodeFilename) {
    std::ifstream file(nodeFilename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << nodeFilename << std::endl;
        return {};
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> number;
    std::vector<glm::vec3> X(number);
    glm::vec3 center(0.0f);

    for (int i = 0; i < number && std::getline(file, line); ++i) {
        std::istringstream lineStream(line);
        int index;
        float x, y, z;
        lineStream >> index >> x >> y >> z;

        X[i].x = x * 0.4f;
        X[i].y = y * 0.4f;
        X[i].z = z * 0.4f;

        center += X[i];
    }

    // Centralize the model
    center /= static_cast<float>(number);
    for (int i = 0; i < number; ++i) {
        X[i] -= center;
        float temp = X[i].y;
        X[i].y = X[i].z;
        X[i].z = temp;
    }

    return X;
}

void SoftBody::Update()
{
    for (int l = 0; l < 10; l++)
        _Update();
}

void SoftBody::_Update()
{
    int threadsPerBlock = 64;
    inspect(inv_Dm, tet_number);
    AddGravity << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, V, mass, number, jump);
    Laplacian_Smoothing();
    glm::vec3 floorPos = glm::vec3(0.0f, -4.0f, 0.0f);
    glm::vec3 floorUp = glm::vec3(0.0f, 1.0f, 0.0f);
    //ComputeForces << <(tet_number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, X, Tet, tet_number, inv_Dm, stiffness_0, stiffness_1);
    UpdateParticles << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (X, V, Force, number, mass, dt, damp, floorPos, floorUp, muT, muN);
}*/


#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "simulation.h"

#include "deformable_mesh.h"
#include "solver.h"

#include <Eigen/Dense>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
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

pd::deformable_mesh_t model{};
pd::solver_t solver;

std::vector<glm::vec3> vertices;
std::vector<GLuint> idx;

int tetNumber = 0;
int vertNumber = 0;

template <typename T>
void inspect(T* dev_ptr, int size) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    std::cout << std::endl;
}

static GuiDataContainer* guiData = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Add the current iteration's output to the overall image
__global__ void AddGravity(glm::vec3* Force, glm::vec3* V, float mass, int numVerts, bool jump)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        if (jump)
            V[index].y += 0.2f;
        Force[index] = glm::vec3(0, -9.8f, 0) * mass;
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
        inv_Dm[index] = Build_Edge_Matrix(X, Tet, index);
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


__global__ void PopulatePos(glm::vec3* vert, glm::vec3* X, GLuint* Tet, int tet_number)
{
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tet < tet_number)
    {
        vert[tet * 12 + 0] = X[Tet[tet * 4 + 0]];
        vert[tet * 12 + 1] = X[Tet[tet * 4 + 2]];
        vert[tet * 12 + 2] = X[Tet[tet * 4 + 1]];
        vert[tet * 12 + 3] = X[Tet[tet * 4 + 0]];
        vert[tet * 12 + 4] = X[Tet[tet * 4 + 3]];
        vert[tet * 12 + 5] = X[Tet[tet * 4 + 2]];
        vert[tet * 12 + 6] = X[Tet[tet * 4 + 0]];
        vert[tet * 12 + 7] = X[Tet[tet * 4 + 1]];
        vert[tet * 12 + 8] = X[Tet[tet * 4 + 3]];
        vert[tet * 12 + 9] = X[Tet[tet * 4 + 1]];
        vert[tet * 12 + 10] = X[Tet[tet * 4 + 2]];
        vert[tet * 12 + 11] = X[Tet[tet * 4 + 3]];
    }
}

void updateVertices(Eigen::MatrixXd positions, int number)
{
    for (int i = 0; i < number; i++)
    {
        vertices[i] = glm::vec3(positions(i, 0), positions(i, 1), positions(i, 2));
    }
}

void initModel()
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
    V.resize(vertNumber, 3);
    for (int i = 0; i < vertNumber; i++)
    {
        V.row(i) = Eigen::Vector3d(
            vertices[i].x,
            vertices[i].y,
            vertices[i].z);
        /**
        V(i, 0) = vertices[i].x;
        V(i, 1) = vertices[i].y;
        V(i, 2) = vertices[i].z;*/
    }

    // allocate space for triangles
    F.resize(tetNumber * 4, 3);
    // triangle indices
    for (int tet = 0; tet < tetNumber; tet++)
    {
        F(4 * tet, 0) = idx[tet * 4 + 0];
        F(4 * tet, 1) = idx[tet * 4 + 2];
        F(4 * tet, 2) = idx[tet * 4 + 1];
        F(4 * tet + 1, 0) = idx[tet * 4 + 0];
        F(4 * tet + 1, 1) = idx[tet * 4 + 3];
        F(4 * tet + 1, 2) = idx[tet * 4 + 2];
        F(4 * tet + 2, 0) = idx[tet * 4 + 0];
        F(4 * tet + 2, 1) = idx[tet * 4 + 1];
        F(4 * tet + 2, 2) = idx[tet * 4 + 3];
        F(4 * tet + 3, 0) = idx[tet * 4 + 1];
        F(4 * tet + 3, 1) = idx[tet * 4 + 2];
        F(4 * tet + 3, 2) = idx[tet * 4 + 3];
    }

    // allocate space for tetrahedra
    T.resize(tetNumber, 4);
    // tet indices
    int a, b, c, d;
    for (int i = 0; i < tetNumber; i++)
    {
        T(i, 0) = idx[i * 4 + 0];
        T(i, 1) = idx[i * 4 + 1];
        T(i, 2) = idx[i * 4 + 2];
        T(i, 3) = idx[i * 4 + 3];
    }
    /**
    for (int i = 0; i < tetNumber; i++)
    {
        cout << T(i, 0) << ", " << T(i, 1) << ", " << T(i, 2) << ", " << T(i, 3) << endl;
        //cout << model.positions()(i, 0) << ", " << model.positions()(i, 1) << ", " << model.positions()(i, 2) << endl;
        //cout << model.mass()(i) << endl;
    }*/
    Eigen::VectorXd masses(V.rows());
    masses.setConstant(10.);
    model = pd::deformable_mesh_t{ V, F, T, masses };
    model.constrain_deformation_gradient(10000.0f);
    //model.velocity().rowwise() += Eigen::RowVector3d{ 0, 0, 0. };
    double const deformation_gradient_wi = 100'000'000.;
    double const positional_wi = 1'000'000'000.;
    model.constrain_deformation_gradient(deformation_gradient_wi);
    
    for (std::size_t i = 0u; i < 10; ++i)
    {
        model.add_positional_constraint(i, positional_wi);
        model.fix(i);
    }
    solver.set_model(&model);
}

void setForce(Eigen::MatrixX3d* fext)
{
    for (std::size_t i = 0; i < vertNumber; ++i)
    {
        double const force = -10000.; // 10 kN
        fext->row(i) += Eigen::RowVector3d{ 0., force, 0. };
    }
}

void pdSolver()
{

    Eigen::MatrixX3d fext;
    fext.resizeLike(model.positions());
    fext.setZero();
    // set gravity force
    fext.col(1).array() -= 900000.0f;
    //setForce(&fext);
    float dt = 0.0163f;
    if (!solver.ready())
    {
        solver.prepare(dt);
    }

    solver.step(fext, 10);

    //std::vector<glm::vec3> vertices(vertNumber);
    updateVertices(model.positions(), vertNumber);
    //fext.setZero();
}

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int number)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < number)
    {
        glm::vec3 v0v1 = X[index * 3 + 2] - X[index * 3 + 1];
        glm::vec3 v0v2 = X[index * 3 + 2] - X[index * 3 + 0];
        glm::vec3 nor = glm::cross(v0v1, v0v2);
        norms[index * 3 + 0] = glm::vec4(nor, 1.f);
        norms[index * 3 + 1] = glm::vec4(nor, 1.f);
        norms[index * 3 + 2] = glm::vec4(nor, 1.f);
    }
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


SimulationCUDAContext::SimulationCUDAContext()
{
}

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto softbody : softBodies) {
        delete softbody;
    }
}

void SimulationCUDAContext::Update()
{
    for (auto softbody : softBodies) {
        softbody->Update();
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->mapDevicePtr(&pos, &nor);
        dim3 numThreadsPerBlock(softbody->getTetNumber() / 32 + 1);
        /**
        glm::vec3* t = (glm::vec3*)malloc(sizeof(glm::vec3) * vertNumber);
        cudaMemcpy(t, softbody->getX(), sizeof(glm::vec3) * vertNumber, cudaMemcpyDeviceToHost);
        for (int i = 0; i < vertNumber; i++)
        {
            cout << t[i].x << "," << t[i].y << "," << t[i].z << endl;
            //cout << model.positions()(i, 0) << ", " << model.positions()(i, 1) << ", " << model.positions()(i, 2) << endl;
            //cout << model.mass()(i) << endl;
        }
        cout << ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,," << endl;*/

        PopulatePos << <numThreadsPerBlock, 32 >> > (pos, softbody->getX(), softbody->getTet(), softbody->getTetNumber());
        RecalculateNormals << <softbody->getNumber() / 32 + 1, 32 >> > (nor, softbody->getX(), softbody->getNumber());
        softbody->unMapDevicePtr();
    }
}

SoftBody::SoftBody(const char* nodeFileName, const char* eleFileName) :Mesh()
{
    vertices = loadNodeFile(nodeFileName);
    number = vertNumber;
    cudaMalloc((void**)&X, sizeof(glm::vec3) * number);
    cudaMemcpy(X, vertices.data(), sizeof(glm::vec3) * number, cudaMemcpyHostToDevice);


    idx = loadEleFile(eleFileName);
    tet_number = tetNumber;
    cudaMalloc((void**)&Tet, sizeof(GLuint) * tetNumber * 4);
    cudaMemcpy(Tet, idx.data(), sizeof(GLuint) * tetNumber * 4, cudaMemcpyHostToDevice);

    Mesh::tet_number = tet_number;

    initModel();

    cudaMalloc((void**)&Force, sizeof(glm::vec3) * number);
    cudaMemset(Force, 0, sizeof(glm::vec3) * number);
    cudaMalloc((void**)&V, sizeof(glm::vec3) * number);
    cudaMemset(V, 0, sizeof(glm::vec3) * number);
    cudaMalloc((void**)&inv_Dm, sizeof(glm::mat4) * tet_number);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * number);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    createTetrahedron();
    cudaMalloc((void**)&V_num, sizeof(int) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDm << < blocks, threadsPerBlock >> > (inv_Dm, tet_number, X, Tet);
}

SoftBody::~SoftBody()
{
    cudaFree(X);
    cudaFree(Tet);
    cudaFree(Force);
    cudaFree(V);
    cudaFree(inv_Dm);
    cudaFree(V_sum);
}

void SoftBody::mapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);

    cudaGraphicsMapResources(1, &cuda_bufNor_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufNorDevPtr, &size, cuda_bufNor_resource);
}

void SoftBody::unMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_bufNor_resource, 0);
}

void SoftBody::Laplacian_Smoothing(float blendAlpha)
{
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    LaplacianGatherKern << < blocks, threadsPerBlock >> > (V, V_sum, V_num, tet_number, Tet);
    LaplacianKern << < (number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, V_sum, V_num, number, Tet, blendAlpha);
}

std::vector<GLuint> SoftBody::loadEleFile(const std::string& EleFilename)
{
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> tetNumber;
    tet_number = tetNumber;
    std::vector<GLuint> Tet(tetNumber * 4);

    int a, b, c, d, e;
    for (int tet = 0; tet < tetNumber && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Tet[tet * 4 + 0] = b - 1;
        Tet[tet * 4 + 1] = c - 1;
        Tet[tet * 4 + 2] = d - 1;
        Tet[tet * 4 + 3] = e - 1;
    }

    file.close();
    return Tet;
}

std::vector<glm::vec3> SoftBody::loadNodeFile(const std::string& nodeFilename) {
    std::ifstream file(nodeFilename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << nodeFilename << std::endl;
        return {};
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> vertNumber;
    number = vertNumber;
    std::vector<glm::vec3> X(number);
    glm::vec3 center(0.0f);

    for (int i = 0; i < number && std::getline(file, line); ++i) {
        std::istringstream lineStream(line);
        int index;
        float x, y, z;
        lineStream >> index >> x >> y >> z;

        X[i].x = x * 0.4f;
        X[i].y = y * 0.4f;
        X[i].z = z * 0.4f;

        center += X[i];
    }

    // Centralize the model
    center /= static_cast<float>(number);
    for (int i = 0; i < number; ++i) {
        X[i] -= center;
        float temp = X[i].y;
        X[i].y = X[i].z;
        X[i].z = temp;
    }

    return X;
}


void SoftBody::Update()
{
    //cout << tetNumber << ", " << tet_number << endl;
    /*
    for (int i = 0; i < vertNumber; i++)
    {
        //cout << idx[i] << endl;
        //cout << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << endl;
        cout << model.positions()(i, 0) << ", " << model.positions()(i, 1) << ", " << model.positions()(i, 2) << endl;
        //cout << model.mass()(i) << endl;
    }
    cout <<",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,," << endl;*/
    pdSolver();
    cudaMalloc((void**)&X, sizeof(glm::vec3) * vertNumber);
    cudaMemcpy(X, vertices.data(), sizeof(glm::vec3) * vertNumber, cudaMemcpyHostToDevice);
}
