#include <cuda.h>

#include <sceneStructs.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <simulationContext.h>
#include <utilities.h>
#include <utilities.cuh>
#include <iostream>
#include <deformable_mesh.h>
#include <solver.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
pd::deformable_mesh_t model{};
pd::solver_t solver;

std::vector<glm::vec3> vertices;
std::vector<GLuint> idx;

void SoftBody::InitModel()
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
    V.resize(number, 3);
    for (int i = 0; i < number; i++)
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
    F.resize(tet_number * 4, 3);
    // triangle indices
    for (int tet = 0; tet < tet_number; tet++)
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
    T.resize(tet_number, 4);
    // tet indices
    int a, b, c, d;
    for (int i = 0; i < tet_number; i++)
    {
        T(i, 0) = idx[i * 4 + 0];
        T(i, 1) = idx[i * 4 + 1];
        T(i, 2) = idx[i * 4 + 2];
        T(i, 3) = idx[i * 4 + 3];
    }

    Eigen::VectorXd masses(V.rows());
    masses.setConstant(mass);
    model = pd::deformable_mesh_t{ V, F, T, masses };
    model.constrain_deformation_gradient(1000000.0f);
    //model.velocity().rowwise() += Eigen::RowVector3d{ 0, 0, 0. };
    double const deformation_gradient_wi = 1000.;
    double const positional_wi = 1'000'000'000.;
    model.constrain_deformation_gradient(deformation_gradient_wi);

    for (std::size_t i = 0u; i < numConstraints; ++i)
    {
        model.add_positional_constraint(i, positional_wi);
        model.fix(i);
    }
    solver.set_model(&model);
}

void SoftBody::PdSolver()
{
    Eigen::MatrixX3d fext;
    fext.resizeLike(model.positions());
    fext.setZero();
    // set gravity force
    fext.col(1).array() -= mpSimContext->GetGravity() * mass;
    //SetForce(&fext);
    if (!solver.ready())
    {
        solver.prepare(mpSimContext->GetDt());
    }

    solver.step(fext, 10);
    //fext.setZero();
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
        RecalculateNormals << <softbody->getTetNumber() * 4 / 32 + 1, 32 >> > (nor, pos, 4 * softbody->getTetNumber());
        softbody->unMapDevicePtr();
    }
}

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
    blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
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
    // ComputeForces << <(tet_number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, X, Tet, tet_number, inv_Dm, stiffness_0, stiffness_1);
    PdSolver();
    positionsFloat = model.positions().cast<float>().transpose();
    cudaMemcpy(X, positionsFloat.data(), number * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    velocitiesFloat = model.velocity().cast<float>();
    cudaMemcpy(V, velocitiesFloat.data(), number * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    HandleFloorCollision << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (X, V, number, floorPos, floorUp, muT, muN);
}