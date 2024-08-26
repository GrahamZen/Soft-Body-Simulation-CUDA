#pragma once

#include <def.h>

struct SolverAttribute {
    float mass = 1.0f;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    int numConstraints = 0;
    bool jump = false;
};

class CollisionDetection;

struct SoftBodyData {
    indexType* Tet = nullptr;
    indexType* Tri = nullptr;
    int numTris = 0;
    int numTets = 0;
};

struct SolverParams {
    struct ExternalForce {
        glm::vec3 jump = glm::vec3(0.f, 400.f, 0.f);
    }extForce;
    SolverAttribute solverAttr;
    float damp = 0.999f;
    float muN = 0.5f;
    float muT = 0.5f;
    float dt = 0.001f;
    float gravity = 9.8f;
    int numIterations = 1;
    bool handleCollision = true;
    CollisionDetection* pCollisionDetection = nullptr;
};

class CollisionDetection;

template<typename HighP>
class Solver {
public:
    Solver(int threadsPerBlock);
    virtual ~Solver();

    virtual void Update(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
    virtual void SolverStep(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
    int threadsPerBlock;

    bool solverReady = false;
};

template<typename HighP>
Solver<HighP>::Solver(int threadsPerBlock) : threadsPerBlock(threadsPerBlock)
{
}

template<typename HighP>
Solver<HighP>::~Solver() {
}