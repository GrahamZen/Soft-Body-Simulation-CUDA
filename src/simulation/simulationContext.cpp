#include <utilities.h>
#include <surfaceshader.h>
#include <projective/pdSolver.h>
#include <IPC/ipc.h>
#include <simulation/simulationContext.h>
#include <simulation/softBody.h>
#include <collision/bvh.h>
#include <context.h>
#include <spdlog/spdlog.h>
#include <map>
#include <chrono>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <limits>

template<class Scalar>
static void CopyUIToParams(const SolverParamsUI& ui, SolverParams<Scalar>& p) {
    p.softBodyAttr = ui.softBodyAttr;
    p.numIterations = ui.numIterations;
    p.maxIterations = ui.maxIterations;
    p.dt = static_cast<Scalar>(ui.dt);
    p.damp = static_cast<Scalar>(ui.damp);
    p.muN = static_cast<Scalar>(ui.muN);
    p.muT = static_cast<Scalar>(ui.muT);
    p.gravity = static_cast<Scalar>(ui.gravity);
    p.dhat = static_cast<Scalar>(ui.dhat);
    p.tol = static_cast<Scalar>(ui.tol);
    if constexpr (std::is_same_v<Scalar, float>) {
        const Scalar minTol = static_cast<Scalar>(1e-6f);
        p.tol = std::max(p.tol, minTol);
    }
    p.rho = static_cast<Scalar>(ui.rho);
    p.handleCollision = ui.handleCollision;
}

template<class Scalar>
static void CopyParamsToUI(const SolverParams<Scalar>& p, SolverParamsUI& ui) {
    ui.softBodyAttr = p.softBodyAttr;
    ui.numIterations = p.numIterations;
    ui.maxIterations = p.maxIterations;
    ui.dt = static_cast<double>(p.dt);
    ui.damp = static_cast<double>(p.damp);
    ui.muN = static_cast<double>(p.muN);
    ui.muT = static_cast<double>(p.muT);
    ui.gravity = static_cast<double>(p.gravity);
    ui.dhat = static_cast<double>(p.dhat);
    ui.tol = static_cast<double>(p.tol);
    ui.rho = static_cast<double>(p.rho);
    ui.handleCollision = p.handleCollision;
}

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>& fixedBodies, int threadsPerBlock, int threadsPerBlockBVH, int maxThreads, int numIterations)
    : threadsPerBlock_(threadsPerBlock), contextGuiData(ctx->guiData), name(_name) {
    std::string prec = "double";
    if (json.contains("precision")) {
        prec = json["precision"].get<std::string>();
    }

    if (prec == "float" || prec == "fp32") {
        precision_ = Precision::Float32;
        auto p = std::make_unique<Impl<float>>();
        p->Init(ctx, json, softBodyDefs, fixedBodies, threadsPerBlock, threadsPerBlockBVH, maxThreads, numIterations);
        CopyParamsToUI(p->params, uiParams_);
        impl_ = std::move(p);
    }
    else {
        precision_ = Precision::Float64;
        auto p = std::make_unique<Impl<double>>();
        p->Init(ctx, json, softBodyDefs, fixedBodies, threadsPerBlock, threadsPerBlockBVH, maxThreads, numIterations);
        CopyParamsToUI(p->params, uiParams_);
        impl_ = std::move(p);
    }
}

SimulationCUDAContext::~SimulationCUDAContext() = default;

void SimulationCUDAContext::Update()
{
    uiParams_.handleCollision = (contextGuiData->handleCollision && contextGuiData->BVHEnabled);

    VisitImpl([&](auto& impl) {
        using Scalar = typename std::decay_t<decltype(impl)>::ScalarType;
        CopyUIToParams(uiParams_, impl.params);
        impl.solver->Update(impl.data, impl.params);
        if (contextGuiData->handleCollision || contextGuiData->BVHEnabled) {
            impl.data.pCollisionDetection->UpdateX(impl.data.X);
            impl.data.pCollisionDetection->PrepareRenderData();
        }
        });
    if (contextGuiData->ObjectVis) {
        PrepareRenderData();
    }
}

void SimulationCUDAContext::SetBVHBuildType(int buildType)
{
    VisitImpl([&](auto& impl) {
        impl.data.pCollisionDetection->SetBuildType(buildType);
        });
}

void SimulationCUDAContext::SetGlobalSolver(int val)
{
    VisitImpl([&](auto& impl) {
        if (auto pdsolver = dynamic_cast<PdSolver*>(impl.solver.get())) {
            pdsolver->SetGlobalSolver(static_cast<PdSolver::SolverType>(val));
        }
        if (auto ipcSolver = dynamic_cast<IPCSolver*>(impl.solver.get())) {
            ipcSolver->SetLinearSolver(static_cast<IPCSolver::SolverType>(val));
        }
        });
}

void SimulationCUDAContext::Draw(SurfaceShader* highLightShaderProgram, SurfaceShader* shaderProgram, SurfaceShader* flatShaderProgram, std::string highLightName)
{
    glLineWidth(2);
    if (!contextGuiData->ObjectVis) return;

    VisitImpl([&](auto& impl) {
        shaderProgram->setModelMatrix(glm::mat4(1.f));
        highLightShaderProgram->setModelMatrix(glm::mat4(1.f));
        for (int i = 0; i < static_cast<int>(impl.softBodies.size()); i++) {
            auto softBody = impl.softBodies[i];
            if (utilityCore::compareHighlightID(softBody->GetName(), highLightName, i))
                highLightShaderProgram->draw(*softBody, 0);
            else
                shaderProgram->draw(*softBody, 0);
        }
        for (int i = 0; i < static_cast<int>(impl.fixedBodies.size()); i++) {
            auto fixedBody = impl.fixedBodies[i];
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
            impl.data.pCollisionDetection->Draw(flatShaderProgram);
        });
}

SolverParamsUI* SimulationCUDAContext::GetSolverParamsUI()
{
    return &uiParams_;
}

void SimulationCUDAContext::SetPerf(bool val)
{
    VisitImpl([&](auto& impl) { impl.solver->SetPerf(val); });
}

const std::vector<std::pair<std::string, double>>& SimulationCUDAContext::GetPerformanceData() const
{
    VisitImpl([&](auto& impl) {
        perfCache_.clear();
        const auto& perf = impl.solver->GetPerformanceData();
        perfCache_.reserve(perf.size());
        for (const auto& p : perf) {
            perfCache_.emplace_back(p.first, static_cast<double>(p.second));
        }
        });
    return perfCache_;
}

MouseSelection SimulationCUDAContext::GetMouseSelection() const
{
    return VisitImpl([&](const auto& impl) { return impl.data.mouseSelection; });
}

void SimulationCUDAContext::SetDragging(bool val)
{
    VisitImpl([&](auto& impl) {
        impl.data.mouseSelection.dragging = val;
        impl.data.mouseSelection.select_v = -1;
        });
}

int SimulationCUDAContext::GetTetCnt() const {
    return VisitImpl([&](const auto& impl) { return impl.data.numTets; });
}

int SimulationCUDAContext::GetVertCnt() const {
    return VisitImpl([&](const auto& impl) { return impl.data.numVerts; });
}

int SimulationCUDAContext::GetNumQueries() const {
    return VisitImpl([&](const auto& impl) { return impl.data.pCollisionDetection->GetNumQueries(); });
}

const std::vector<SoftBody*>& SimulationCUDAContext::GetSoftBodies() const {
    return VisitImpl([&](const auto& impl) -> const std::vector<SoftBody*>&{ return impl.softBodies; });
}

const std::vector<FixedBody*>& SimulationCUDAContext::GetFixedBodies() const {
    return VisitImpl([&](const auto& impl) -> const std::vector<FixedBody*>&{ return impl.fixedBodies; });
}
