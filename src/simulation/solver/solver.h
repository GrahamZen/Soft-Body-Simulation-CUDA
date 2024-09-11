#pragma once

#include <def.h>

struct SoftBodyAttribute {
    float mass = 1.0f;
    float mu = 20000.0f;
    float lambda = 5000.0f;
    indexType* DBC = nullptr;
    size_t numDBC = 0;
    bool jump = false;
};

struct SoftBodyData {
    indexType* Tri = nullptr;
    int numTris = 0;
};

template<typename Scalar>
struct SolverParams {
    struct ExternalForce {
        glm::vec3 jump = glm::vec3(0.f, 400.f, 0.f);
    }extForce;
    SoftBodyAttribute softBodyAttr;
    float damp = 0.999f;
    float muN = 0.5f;
    float muT = 0.5f;
    float dt = 0.001f;
    float gravity = 9.8f;
    int numIterations = 1;
    bool handleCollision = true;
};

template<typename Scalar>
class Solver {
public:
    Solver(int threadsPerBlock);
    virtual ~Solver();

    virtual void Update(SolverData<Scalar>& solverData, SolverParams<Scalar>& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData<Scalar>& solverData, SolverParams<Scalar>& solverParams) = 0;
    virtual void SolverStep(SolverData<Scalar>& solverData, SolverParams<Scalar>& solverParams) = 0;
    int threadsPerBlock;

    bool solverReady = false;
};

template<typename Scalar>
Solver<Scalar>::Solver(int threadsPerBlock) : threadsPerBlock(threadsPerBlock)
{
}

template<typename Scalar>
Solver<Scalar>::~Solver() {
}