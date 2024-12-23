#include <utilities.h>
#include <surfaceshader.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/simulationContext.h>
#include <simulation/softBody.h>
#include <collision/bvh.h>
#include <context.h>
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
    mSolverParams.handleCollision = (contextGuiData->handleCollision && contextGuiData->BVHEnabled);
    mSolver->Update(mSolverData, mSolverParams);
    if (contextGuiData->handleCollision || contextGuiData->BVHEnabled) {
        mSolverData.pCollisionDetection->UpdateX(mSolverData.X);
        mSolverData.pCollisionDetection->PrepareRenderData();
    }
    if (contextGuiData->ObjectVis) {
        PrepareRenderData();
    }
}

void SimulationCUDAContext::SetBVHBuildType(int buildType)
{
    mSolverData.pCollisionDetection->SetBuildType(buildType);
}

void SimulationCUDAContext::SetGlobalSolver(int val)
{
    PdSolver* pdsolver = nullptr;
    if (pdsolver = dynamic_cast<PdSolver*>(mSolver)) {
        pdsolver->SetGlobalSolver(static_cast<PdSolver::SolverType>(val));
    }
}

void SimulationCUDAContext::Reset()
{
    cudaMemcpy(mSolverData.X, mSolverData.X0, sizeof(glm::tvec3<solverPrecision>) * mSolverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(mSolverData.XTilde, mSolverData.X0, sizeof(glm::tvec3<solverPrecision>) * mSolverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemset(mSolverData.V, 0, sizeof(glm::tvec3<solverPrecision>) * mSolverData.numVerts);
    mSolver->Reset();
}

void SimulationCUDAContext::Draw(SurfaceShader* highLightShaderProgram, SurfaceShader* shaderProgram, SurfaceShader* flatShaderProgram, std::string highLightName)
{
    glLineWidth(2);
    if (contextGuiData->ObjectVis) {
        shaderProgram->setModelMatrix(glm::mat4(1.f));
        highLightShaderProgram->setModelMatrix(glm::mat4(1.f));
        for (int i = 0; i < softBodies.size(); i++) {
            auto softBody = softBodies[i];
            if (utilityCore::compareHighlightID(softBody->GetName(), highLightName, i))
                highLightShaderProgram->draw(*softBody, 0);
            else
                shaderProgram->draw(*softBody, 0);
        }
        for (int i = 0; i < fixedBodies.size(); i++) {
            auto fixedBody = fixedBodies[i];
            if (utilityCore::compareHighlightID(fixedBody->name, highLightName, i)) {
                highLightShaderProgram->setModelMatrix(fixedBody->m_model);
                highLightShaderProgram->draw(*fixedBody, 0);
            }
            else {
                shaderProgram->setModelMatrix(fixedBody->m_model);
                shaderProgram->draw(*fixedBody, 0);
            }
        }
        if (contextGuiData->handleCollision && contextGuiData->BVHEnabled)
            mSolverData.pCollisionDetection->Draw(flatShaderProgram);
    }
}

SolverParams<solverPrecision>* SimulationCUDAContext::GetSolverParams()
{
    return &mSolverParams;
}

void SimulationCUDAContext::SetPerf(bool val)
{
    mSolver->SetPerf(val);
}

const std::vector<std::pair<std::string, solverPrecision>>& SimulationCUDAContext::GetPerformanceData() const
{
    return mSolver->GetPerformanceData();
}
