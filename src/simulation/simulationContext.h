#pragma once

#include <fixedBodyData.h>
#include <simulation/solver/solver.h>
#include <json.hpp>

class SoftBody;
class SurfaceShader;
class Context;
class SoftBodyAttr;
using solverPrecision = double;

class SimulationCUDAContext {
    template<typename Scalar>
    friend class CollisionDetection;
public:
    SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
        const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>&, int threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations);
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    const std::vector<const char*>& GetNamesSoftBodies() const { return namesSoftBodies; }
    void UpdateSingleSBAttr(int index, SoftBodyAttr* pSoftBodyAttr);
    void SetBVHBuildType(int);
    void SetGlobalSolver(bool useEigen);
    void Draw(SurfaceShader*, SurfaceShader*);
    SolverParams<solverPrecision>* GetSolverParams();
    int GetTetCnt() const;
    int GetVertCnt() const;
    int GetThreadsPerBlock() const { return threadsPerBlock; }
    int GetNumQueries() const;
private:
    void PrepareRenderData();
    int threadsPerBlock = 64;
    SolverData<solverPrecision> mSolverData;
    std::vector<const char*> namesSoftBodies;
    std::vector<SoftBody*> softBodies;
    std::vector<FixedBody*> fixedBodies;
    std::vector<int> startIndices;

    Context* context = nullptr;
    const std::string name;
    SolverParams<solverPrecision> mSolverParams;
    Solver<solverPrecision>* mSolver = nullptr;
};