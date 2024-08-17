#pragma once

#include <def.h>

struct SolverAttribute {
    float mass = 1.0f;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    int numConstraints = 0;
    bool jump = false;
};

class SimulationCUDAContext;

struct SolverData {
    indexType* Tet = nullptr;
    indexType* Tri = nullptr;
    glm::vec3* Force = nullptr;
    glm::vec3* V = nullptr;
    glm::vec3* X = nullptr;
    glm::vec3* X0 = nullptr;
    glm::vec3* XTilt = nullptr;
    glm::vec3* dev_ExtForce = nullptr;
    glm::mat3* inv_Dm = nullptr;
    float* V0 = nullptr;
    int numTets = 0;
    int numVerts = 0;
    int numTris = 0;
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
    int numIterations = 10;
};

class Solver {
public:
    Solver();
    virtual ~Solver();

    virtual void Update(SolverData& solverData, SolverParams& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverParams& solverParams) = 0;
    virtual void SolverStep(SolverData& solverData, SolverParams& solverParams) = 0;
    int threadsPerBlock;

    bool solverReady = false;
};
