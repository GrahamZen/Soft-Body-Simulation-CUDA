#pragma once

#include <fixedBodyData.h>
#include <simulation/solver/solver.h>
#include <precision.h>
#include <json.hpp>

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
    void Reset();
    void UpdateSoftBodyAttr(int index, SoftBodyAttr* pSoftBodyAttr);
    void SetBVHBuildType(int);
    void SetGlobalSolver(int val);
    void Draw(SurfaceShader*, SurfaceShader*, SurfaceShader*, std::string highLightName = "");
    SolverParams<solverPrecision>* GetSolverParams();
    void SetPerf(bool val);
    const std::vector<std::pair<std::string, solverPrecision>>& GetPerformanceData() const;
    int GetTetCnt() const;
    int GetVertCnt() const;
    int GetThreadsPerBlock() const { return threadsPerBlock; }
    int GetNumQueries() const;
    const std::vector<SoftBody*>& GetSoftBodies() const { return softBodies; }
    const std::vector<FixedBody*>& GetFixedBodies() const { return fixedBodies; }
    std::string GetName() const { return name; }
private:
    void PrepareRenderData();
    int threadsPerBlock = 64;
    SolverData<solverPrecision> mSolverData;
    std::vector<SoftBody*> softBodies;
    std::vector<FixedBody*> fixedBodies;
    std::vector<int> startIndices;

    GuiDataContainer* contextGuiData = nullptr;
    const std::string name;
    SolverParams<solverPrecision> mSolverParams;
    Solver<solverPrecision>* mSolver = nullptr;
};