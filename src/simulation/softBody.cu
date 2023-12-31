#include <simulationContext.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <utilities.cuh>

SoftBody::SoftBody(SimulationCUDAContext* context, SoftBodyAttribute& _attrib, SoftBodyData* dataPtr)
    : mcrpSimContext(context), threadsPerBlock(context->GetThreadsPerBlock()), attrib(_attrib), Tet(dataPtr->Tets), Tri(dataPtr->Tris), X(dataPtr->dev_X), X0(dataPtr->dev_X0), XTilt(dataPtr->dev_XTilt),
    V(dataPtr->dev_V), Force(dataPtr->dev_F), numTets(dataPtr->numTets), numVerts(dataPtr->numVerts), numTris(dataPtr->numTris), dev_ExtForce(numVerts, glm::vec3(0.f))
{
    vertices.resize(numVerts);
    cudaMemcpy(vertices.data(), X, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToHost);
    idx.resize(numTets * 4);
    cudaMemcpy(idx.data(), Tet, sizeof(int) * numTets * 4, cudaMemcpyDeviceToHost);

    Mesh::numTets = numTets;
    Mesh::numTris = numTris;

    if (numTris == 0)
        createTetrahedron();
    else
        createMesh();
    cudaMalloc((void**)&inv_Dm, sizeof(glm::mat4) * numTets);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * numVerts);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * numVerts);
    cudaMalloc((void**)&V_num, sizeof(int) * numVerts);
    cudaMemset(V_num, 0, sizeof(int) * numVerts);
    cudaMalloc((void**)&V0, sizeof(float) * numTets);
    cudaMemset(V0, 0, sizeof(float) * numTets);
    int blocks = (numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (V0, inv_Dm, numTets, X, Tet);
}

SoftBody::~SoftBody()
{
    cudaFree(Tet);
    cudaFree(Force);
    cudaFree(V);
    cudaFree(inv_Dm);
    cudaFree(V_sum);

    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);

    free(bHost);
    cudaFree(buffer_gpu);
}


void SoftBody::PDSolver()
{
    if (!solverReady)
    {
        solverPrepare();
        solverReady = true;
    }
    PDSolverStep();
}


void SoftBody::Laplacian_Smoothing(float blendAlpha)
{
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * numVerts);
    cudaMemset(V_num, 0, sizeof(int) * numVerts);
    int blocks = (numTets + threadsPerBlock - 1) / threadsPerBlock;
    LaplacianGatherKern << < blocks, threadsPerBlock >> > (V, V_sum, V_num, numTets, Tet);
    LaplacianKern << < (numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, V_sum, V_num, numVerts, Tet, blendAlpha);
}

void SoftBody::Update()
{
    _Update();
}

void SoftBody::Reset()
{
    cudaMemset(Force, 0, sizeof(glm::vec3) * numVerts);
    cudaMemset(V, 0, sizeof(glm::vec3) * numVerts);
    cudaMemcpy(X, X0, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(XTilt, X0, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToDevice);
}

void SoftBody::_Update()
{
    AddExternal << <(numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, numVerts, jump, attrib.mass, mcrpSimContext->GetExtForce().jump);
    // Laplacian_Smoothing();
    //ComputeForces << <(numTets + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, X, Tet, numTets, inv_Dm, stiffness_0, stiffness_1);
    PDSolver();
}