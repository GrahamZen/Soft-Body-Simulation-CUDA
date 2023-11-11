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

//Kernel that writes the image to the OpenGL VBO directly.
__global__ void sendImageToVBO(uchar4* vbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        vbo[index].w = 0;
        vbo[index].x = color.x;
        vbo[index].y = color.y;
        vbo[index].z = color.z;
    }
}

static GuiDataContainer* guiData = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Add the current iteration's output to the overall image
__global__ void Step(glm::vec3* X, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        X[index].y += 0.001f;
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
        glm::vec3 v0v1 = X[number * 3 + 2] - X[number * 3 + 1];
        glm::vec3 v0v2 = X[number * 3 + 2] - X[number * 3 + 0];
        glm::vec3 nor = glm::cross(v0v1, v0v2);
        norms[number * 3 + 0] = glm::vec4(nor, 1.f);
        norms[number * 3 + 1] = glm::vec4(nor, 1.f);
        norms[number * 3 + 2] = glm::vec4(nor, 1.f);
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
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
    createTetrahedron();
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
    Step << <number / 32 + 1, 32 >> > (X, number);
}

void SoftBody::_Update()
{
}
