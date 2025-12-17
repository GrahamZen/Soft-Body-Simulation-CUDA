#pragma once

#include <fixedBodyData.h>
#include <simulation/solver/solver.h>
#include <json.hpp>
#include <sceneStructs.h>
#include <memory>
#include <variant>
#include <vector>

class SoftBody;
class SurfaceShader;
class Context;
class GuiDataContainer;
class SoftBodyAttr;

class SimulationCUDAContext {
public:
    SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
        const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>&, int threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations);
    ~SimulationCUDAContext();
    void Update();
    void UpdateDBC();
    void Reset();
    void UpdateSoftBodyAttr(int index, SoftBodyAttr* pSoftBodyAttr);
    void SetBVHBuildType(int);
    void SetGlobalSolver(int val);
    void ResetMoreDBC(bool clear = false);
    bool RayIntersect(const Ray& ray, glm::vec3* pos, bool updateV = true);
    void Draw(SurfaceShader*, SurfaceShader*, SurfaceShader*, std::string highLightName = "");
    SolverParamsUI* GetSolverParamsUI();
    void SetPerf(bool val);
    const std::vector<std::pair<std::string, double>>& GetPerformanceData() const;
    MouseSelection GetMouseSelection() const;
    void SetDragging(bool val);
    int GetTetCnt() const;
    int GetVertCnt() const;
    int GetThreadsPerBlock() const { return threadsPerBlock_; }
    int GetNumQueries() const;
    const std::vector<SoftBody*>& GetSoftBodies() const;
    const std::vector<FixedBody*>& GetFixedBodies() const;
    std::string GetName() const { return name; }

    Precision GetPrecision() const { return precision_; }
private:
    void PrepareRenderData();

    template<class Scalar>
    struct Impl {
        using ScalarType = Scalar;

        int threadsPerBlock = 64;
        int threadsPerBlockBVH = 64;
        int maxThreads = 0;
        int numIterations = 10;

        SolverData<Scalar> data{};
        SolverParams<Scalar> params{};
        std::unique_ptr<Solver<Scalar>> solver;

        std::vector<SoftBody*> softBodies;
        std::vector<FixedBody*> fixedBodies;
        std::vector<int> startIndices;

        void Init(Context* ctx, nlohmann::json& json,
            const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>&,
            int threadsPerBlock, int threadsPerBlockBVH, int maxThreads, int numIterations);
        ~Impl();
    };

    template<class F>
    decltype(auto) VisitImpl(F&& f);
    template<class F>
    decltype(auto) VisitImpl(F&& f) const;

    int threadsPerBlock_ = 64;
    GuiDataContainer* contextGuiData = nullptr;
    const std::string name;

    Precision precision_ = Precision::Float64;
    SolverParamsUI uiParams_;
    mutable std::vector<std::pair<std::string, double>> perfCache_;

    std::variant<std::unique_ptr<Impl<float>>, std::unique_ptr<Impl<double>>> impl_;
};

extern template struct SimulationCUDAContext::Impl<float>;
extern template struct SimulationCUDAContext::Impl<double>;

template<class F>
inline decltype(auto) SimulationCUDAContext::VisitImpl(F&& f) {
    return std::visit([&](auto& p) -> decltype(auto) { return f(*p); }, impl_);
}
template<class F>
inline decltype(auto) SimulationCUDAContext::VisitImpl(F&& f) const {
    return std::visit([&](auto& p) -> decltype(auto) { return f(*p); }, impl_);
}
