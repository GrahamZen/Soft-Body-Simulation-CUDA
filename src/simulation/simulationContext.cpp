#include <utilities.h>
#include <surfaceshader.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/simulationContext.h>
#include <simulation/softBody.h>
#include <spdlog/spdlog.h>
#include <map>
#include <chrono>
#include <functional>

template<typename Func>
void measureExecutionTime(const Func& func, const std::string& message, bool print = false) {
    if (!print) {
        func();
        return;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("{} Time: {} milliseconds", message, milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void SimulationCUDAContext::Update()
{
    mSolverParams.handleCollision = (context->guiData->handleCollision && context->guiData->BVHEnabled);
    mSolver->Update(mSolverData, mSolverParams);
    if (context->guiData->handleCollision || context->guiData->BVHEnabled) {
        mSolverParams.pCollisionDetection->PrepareRenderData();
    }
    if (context->guiData->ObjectVis) {
        PrepareRenderData();
    }
}


void SimulationCUDAContext::UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr) {
    softBodies[index]->SetAttributes(softBodyAttr);
}

void SimulationCUDAContext::SetBVHBuildType(BVH<solverPrecision>::BuildType buildType)
{
    mSolverParams.pCollisionDetection->SetBuildType(buildType);
}

void SimulationCUDAContext::SetGlobalSolver(bool useEigen)
{
    PdSolver* pdsolver = nullptr;
    if (pdsolver = dynamic_cast<PdSolver*>(mSolver)) {
        pdsolver->SetGlobalSolver(useEigen);
    }
}

void SimulationCUDAContext::Reset()
{
    cudaMemcpy(mSolverData.X, mSolverData.X0, sizeof(glm::tvec3<solverPrecision>) * mSolverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(mSolverData.XTilde, mSolverData.X0, sizeof(glm::tvec3<solverPrecision>) * mSolverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemset(mSolverData.V, 0, sizeof(glm::tvec3<solverPrecision>) * mSolverData.numVerts);
}

void SimulationCUDAContext::Draw(SurfaceShader* shaderProgram, SurfaceShader* flatShaderProgram)
{
    glLineWidth(2);
    if (context->guiData->ObjectVis) {
        shaderProgram->setModelMatrix(glm::mat4(1.f));
        for (auto softBody : softBodies)
            shaderProgram->draw(*softBody, 0);
        for (auto fixedBody : fixedBodies) {
            shaderProgram->setModelMatrix(fixedBody->m_model);
            shaderProgram->draw(*fixedBody, 0);
        }
    }
    if (context->guiData->handleCollision && context->guiData->BVHEnabled) 
        mSolverParams.pCollisionDetection->Draw(flatShaderProgram);
}

const SolverParams<solverPrecision>& SimulationCUDAContext::GetSolverParams() const
{
    return mSolverParams;
}
