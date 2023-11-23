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
    number = vertices.size();
    cudaMalloc((void**)&X, sizeof(glm::vec3) * number);
    cudaMemcpy(X, vertices.data(), sizeof(glm::vec3) * number, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int threadsPerBlock = 64;
    int blocks = (number + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (X, model, number);
    cudaMemcpy(vertices.data(), X, sizeof(glm::vec3) * number, cudaMemcpyDeviceToHost);

    cudaMalloc((void**)&X0, sizeof(glm::vec3) * number);
    cudaMemcpy(X0, X, sizeof(glm::vec3) * number, cudaMemcpyDeviceToDevice);

    idx = loadEleFile(eleFileName, startIndex);
    tet_number = idx.size() / 4;
    cudaMalloc((void**)&Tet, sizeof(GLuint) * idx.size());
    cudaMemcpy(Tet, idx.data(), sizeof(GLuint) * idx.size(), cudaMemcpyHostToDevice);

    Mesh::tet_number = tet_number;

    InitModel();

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
    cudaMalloc((void**)&V0, sizeof(float) * tet_number);
    cudaMemset(V0, 0, sizeof(float) * tet_number);
    blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (V0, inv_Dm, tet_number, X, Tet);
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
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    LaplacianGatherKern << < blocks, threadsPerBlock >> > (V, V_sum, V_num, tet_number, Tet);
    LaplacianKern << < (number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, V_sum, V_num, number, Tet, blendAlpha);
}

void SoftBody::Update()
{
    _Update();
}

void SoftBody::Reset()
{
    cudaMemset(Force, 0, sizeof(glm::vec3) * number);
    cudaMemset(V, 0, sizeof(glm::vec3) * number);
    cudaMemcpy(X, X0, sizeof(glm::vec3) * number, cudaMemcpyDeviceToDevice);
    InitModel();
}

void SoftBody::_Update()
{
    int threadsPerBlock = 64;
    AddGravity << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, V, mass, number, jump);
    Eigen::MatrixXf positionsFloat;
    using RowMajorMatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    RowMajorMatrixX3f velocitiesFloat(number, 3);
    positionsFloat.resizeLike(model.positions().transpose());
    cudaMemcpy(positionsFloat.data(), X, number * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocitiesFloat.data(), V, number * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    model.positions() = positionsFloat.transpose().cast<double>();
    model.velocity() = velocitiesFloat.cast<double>();
    // Laplacian_Smoothing();
    glm::vec3 floorPos = glm::vec3(0.0f, -4.0f, 0.0f);
    glm::vec3 floorUp = glm::vec3(0.0f, 1.0f, 0.0f);
    //ComputeForces << <(tet_number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, X, Tet, tet_number, inv_Dm, stiffness_0, stiffness_1);
    if (useGPUSolver)
    {
        PDSolver();
    }
    else
    {
        PdSolver();
        positionsFloat = model.positions().cast<float>().transpose();
        cudaMemcpy(X, positionsFloat.data(), number * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        velocitiesFloat = model.velocity().cast<float>();
        cudaMemcpy(V, velocitiesFloat.data(), number * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    HandleFloorCollision << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (X, V, number, floorPos, floorUp, muT, muN);
}