#pragma once

#include <def.h>
#include <thrust/device_vector.h>

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

class Solver {
public:
    Solver(SimulationCUDAContext*);
    virtual ~Solver();

    virtual void Update(SolverData& solverData, SolverAttribute& solverAttr) = 0;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverAttribute& solverAttr) = 0;
    virtual void SolverStep(SolverData& solverData, SolverAttribute& solverAttr) = 0;
    const int threadsPerBlock;
    SimulationCUDAContext* mcrpSimContext;

    bool solverReady = false;
};
