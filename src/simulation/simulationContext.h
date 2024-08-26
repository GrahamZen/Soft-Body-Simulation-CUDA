#pragma once

#include <collision/bvh.h>
#include <fixedBodyData.h>
#include <simulation/solver/solver.h>
#include <context.h>
#include <json.hpp>

class SoftBody;
class SurfaceShader;

class SimulationCUDAContext {
    friend class CollisionDetection;
public:
    SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
        const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>&, int threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations);
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    const std::vector<const char*>& GetNamesSoftBodies() const { return namesSoftBodies; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { mSolverParams.dt = dt; }
    void SetBVHBuildType(BVH::BuildType);
    void SetGlobalSolver(bool useEigen);
    void Draw(SurfaceShader*, SurfaceShader*);
    const SolverParams& GetSolverParams() const;
    int GetTetCnt() const;
    int GetVertCnt() const;
    int GetThreadsPerBlock() const { return threadsPerBlock; }
    int GetNumQueries() const;
private:
    void PrepareRenderData();
    int threadsPerBlock = 64;
    SolverData<double> mSolverData;
    indexType* dev_TetFathers;
    indexType* dev_Edges;
    std::vector<const char*> namesSoftBodies;
    std::vector<SoftBody*> softBodies;
    std::vector<FixedBody*> fixedBodies;
    std::vector<int> startIndices;

    Context* context = nullptr;
    const std::string name;
    SolverParams mSolverParams;
    Solver<double>* mSolver = nullptr;
};