#include <cuda.h>

#include <sceneStructs.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <simulationContext.h>
#include <utilities.h>
#include <utilities.cuh>
#include <iostream>
#include <cusolverSp.h>
#include <cusparse.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

std::vector<Eigen::Triplet<float>> A_triplets;

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
    model.constrain_deformation_gradient(wi);
    //model.velocity().rowwise() += Eigen::RowVector3d{ 0, 0, 0. };
    double const positional_wi = 1'000'000'000.;
    //model.constrain_deformation_gradient(deformation_gradient_wi);

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
    if (!solver.ready())
    {
        solver.prepare(mpSimContext->GetDt());
    }
    solver.step(fext, 10);
    //fext.setZero();
}

void SoftBody::solverPrepare()
{
    int threadsPerBlock = 64;
    int vertBlocks = (number + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    float dt = mpSimContext->GetDt();
    float const m_1_dt2 = mass / (dt * dt);
    int len = number * 3 + 48 * tet_number;
    int ASize = 3 * number;

    cudaMalloc((void**)&sn, sizeof(float) * ASize);
    cudaMalloc((void**)&b, sizeof(float) * ASize);
    cudaMalloc((void**)&masses, sizeof(float) * ASize);

    int* AIdx;
    cudaMalloc((void**)&AIdx, sizeof(int) * len);
    cudaMemset(AIdx, 0, sizeof(int) * len);

    float* tmpVal;
    cudaMalloc((void**)&tmpVal, sizeof(int) * len);
    cudaMemset(tmpVal, 0, sizeof(int) * len);

    cudaMalloc((void**)&ExtForce, sizeof(glm::vec3) * number);
    cudaMemset(ExtForce, 0, sizeof(float) * number);

    computeSiTSi << < tetBlocks, threadsPerBlock >> > (AIdx, tmpVal, V0, inv_Dm, Tet, wi, tet_number, number);
    setMDt_2 << < vertBlocks, threadsPerBlock >> > (AIdx, tmpVal, 48 * tet_number, m_1_dt2, number);


    if (useEigen)
    {
        bHost = (float*)malloc(sizeof(float) * ASize);

        int* AIdxHost = (int*)malloc(sizeof(int) * len);
        float* tmpValHost = (float*)malloc(sizeof(float) * len);

        cudaMemcpy(AIdxHost, AIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmpValHost, tmpVal, sizeof(float) * len, cudaMemcpyDeviceToHost);

        std::vector<Eigen::Triplet<float>> A_triplets;

        for (auto i = 0; i < len; ++i)
        {
            A_triplets.push_back({ AIdxHost[i] / ASize, AIdxHost[i] % ASize, tmpValHost[i] });
        }
        Eigen::SparseMatrix<float> A(ASize, ASize);

        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
        cholesky_decomposition_.compute(A);

        free(AIdxHost);
        free(tmpValHost);
    }
    else
    {
        int* newIdx;
        float* newVal;

        cudaMalloc((void**)&newIdx, sizeof(int) * len);
        cudaMalloc((void**)&newVal, sizeof(float) * len);

        thrust::sort_by_key(thrust::device, AIdx, AIdx + len, tmpVal);


        thrust::pair<int*, float*> newEnd = thrust::reduce_by_key(thrust::device, AIdx, AIdx + len, tmpVal, newIdx, newVal);

        nnzNumber = newEnd.first - newIdx;
        std::cout << nnzNumber << std::endl;

        cudaMalloc((void**)&ARow, sizeof(int) * nnzNumber);
        cudaMemset(ARow, 0, sizeof(int) * nnzNumber);

        cudaMalloc((void**)&ACol, sizeof(int) * nnzNumber);
        cudaMemset(ACol, 0, sizeof(int) * nnzNumber);

        cudaMalloc((void**)&AVal, sizeof(float) * nnzNumber);
        cudaMemcpy(AVal, newVal, sizeof(float) * nnzNumber, cudaMemcpyDeviceToDevice);

        int* ARowTmp;
        cudaMalloc((void**)&ARowTmp, sizeof(int) * nnzNumber);
        cudaMemset(ARowTmp, 0, sizeof(int) * nnzNumber);

        //int threadsPerBlock = 64;
        int blocks = (nnzNumber + threadsPerBlock - 1) / threadsPerBlock;

        initAMatrix << < blocks, threadsPerBlock >> > (newIdx, ARowTmp, ACol, ASize, nnzNumber);

        // transform ARow into csr format
        cusparseHandle_t handle;
        cusparseCreate(&handle);
        cusparseXcoo2csr(handle, ARowTmp, nnzNumber, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

        cudaFree(newIdx);
        cudaFree(newVal);
        cudaFree(ARowTmp);
    }
    cudaFree(AIdx);
    cudaFree(tmpVal);
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

void SoftBody::PDSolverStep()
{

    float dt = mpSimContext->GetDt();
    float const dtInv = 1.0f / dt;
    float const dt2 = dt * dt;
    float const dt2_m_1 = dt2 / mass;
    float const m_1_dt2 = mass / dt2;


    int threadsPerBlock = 64;
    int vertBlocks = (number + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;

    glm::vec3 gravity = glm::vec3(0.0f, -mpSimContext->GetGravity(), 0.0f);
    setExtForce << < vertBlocks, threadsPerBlock >> > (ExtForce, gravity, number);
    computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, X, V, ExtForce, number);
    computeM_h2Sn << < vertBlocks, threadsPerBlock >> > (masses, sn, m_1_dt2, number);
    cusolverSpHandle_t cusolverHandle;
    int singularity = 0;
    cusparseMatDescr_t descrA;
    if (!useEigen)
    {
        cusolverSpCreate(&cusolverHandle);
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    }

    // 10 is the number of iterations
    for (int i = 0; i < 10; i++)
    {
        cudaMemset(b, 0, sizeof(float) * number * 3);
        computeLocal << < tetBlocks, threadsPerBlock >> > (V0, wi, b, inv_Dm, sn, Tet, tet_number);
        addM_h2Sn << < vertBlocks, threadsPerBlock >> > (b, masses, number);

        if (useEigen)
        {
            cudaMemcpy(bHost, b, sizeof(float) * (number * 3), cudaMemcpyDeviceToHost);
            Eigen::VectorXf bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, number * 3);
            Eigen::VectorXf res = cholesky_decomposition_.solve(bh);
            cudaMemcpy(sn, res.data(), sizeof(float) * (number * 3), cudaMemcpyHostToDevice);
        }
        else
        {
            cusolverSpScsrlsvchol(cusolverHandle, number * 3, nnzNumber, descrA, AVal, ARow, ACol, b, 0.0001f, 0, sn, &singularity);
        }
    }

    updateVelPos << < vertBlocks, threadsPerBlock >> > (sn, dtInv, X, V, number);
}

void SimulationCUDAContext::Update()
{
    //m_bvh.BuildBVHTree(0, GetAABB(), GetTetCnt(), softBodies);
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