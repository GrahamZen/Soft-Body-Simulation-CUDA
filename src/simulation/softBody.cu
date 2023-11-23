#include <simulationContext.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <utilities.cuh>

SoftBody::SoftBody(const char* nodeFileName, const char* eleFileName, SimulationCUDAContext* context, const glm::vec3& pos, const glm::vec3& scale,
    const glm::vec3& rot, float mass, float stiffness_0, float stiffness_1, float damp, float muN, float muT, int constraints, bool centralize, int startIndex)
    : mpSimContext(context), mass(mass), stiffness_0(stiffness_0), stiffness_1(stiffness_1), damp(damp), muN(muN), muT(muT), numConstraints(constraints)
{
    vertices = loadNodeFile(nodeFileName, centralize);
    numVerts = vertices.size();
    cudaMalloc((void**)&X, sizeof(glm::vec3) * numVerts);
    cudaMemcpy(X, vertices.data(), sizeof(glm::vec3) * numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int threadsPerBlock = 64;
    int blocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (X, model, numVerts);
    cudaMemcpy(vertices.data(), X, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToHost);

    cudaMalloc((void**)&X0, sizeof(glm::vec3) * numVerts);
    cudaMemcpy(X0, X, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToDevice);

    idx = loadEleFile(eleFileName, startIndex);
    numTets = idx.size() / 4;
    cudaMalloc((void**)&Tet, sizeof(GLuint) * idx.size());
    cudaMemcpy(Tet, idx.data(), sizeof(GLuint) * idx.size(), cudaMemcpyHostToDevice);

    Mesh::numTets = numTets;

    InitModel();

    cudaMalloc((void**)&Force, sizeof(glm::vec3) * numVerts);
    cudaMemset(Force, 0, sizeof(glm::vec3) * numVerts);
    cudaMalloc((void**)&V, sizeof(glm::vec3) * numVerts);
    cudaMemset(V, 0, sizeof(glm::vec3) * numVerts);
    cudaMalloc((void**)&inv_Dm, sizeof(glm::mat4) * numTets);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * numVerts);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * numVerts);
    createTetrahedron();
    cudaMalloc((void**)&V_num, sizeof(int) * numVerts);
    cudaMemset(V_num, 0, sizeof(int) * numVerts);
    cudaMalloc((void**)&V0, sizeof(float) * numTets);
    cudaMemset(V0, 0, sizeof(float) * numTets);
    blocks = (numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (V0, inv_Dm, numTets, X, Tet);
}

SoftBody::~SoftBody()
{
    cudaFree(X);
    cudaFree(Tet);
    cudaFree(Force);
    cudaFree(V);
    cudaFree(inv_Dm);
    cudaFree(V_sum);

    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);

    if (useEigen)
    {
        free(bHost);
    }
    else
    {
        cudaFree(ARow);
        cudaFree(ACol);
        cudaFree(AVal);
    }
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
    int threadsPerBlock = 64;
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
    InitModel();
}

void SoftBody::_Update()
{
    int threadsPerBlock = 64;
    AddGravity << <(numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, V, mass, numVerts, jump);
    // Laplacian_Smoothing();
    glm::vec3 floorPos = glm::vec3(0.0f, -4.0f, 0.0f);
    glm::vec3 floorUp = glm::vec3(0.0f, 1.0f, 0.0f);
    //ComputeForces << <(numTets + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, X, Tet, numTets, inv_Dm, stiffness_0, stiffness_1);
    if (useGPUSolver)
    {
        PDSolver();
    }
    else
    {
        Eigen::MatrixXf positionsFloat;
        using RowMajorMatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
        RowMajorMatrixX3f velocitiesFloat(numVerts, 3);
        positionsFloat.resizeLike(model.positions().transpose());
        cudaMemcpy(positionsFloat.data(), X, numVerts * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocitiesFloat.data(), V, numVerts * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        model.positions() = positionsFloat.transpose().cast<double>();
        model.velocity() = velocitiesFloat.cast<double>();
        PdSolver();
        positionsFloat = model.positions().cast<float>().transpose();
        cudaMemcpy(X, positionsFloat.data(), numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        velocitiesFloat = model.velocity().cast<float>();
        cudaMemcpy(V, velocitiesFloat.data(), numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    HandleFloorCollision << <(numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (X, V, numVerts, floorPos, floorUp, muT, muN);
}