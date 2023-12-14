#pragma once

#include <def.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

struct SolverAttribute {
    float mass = 1.0f;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    int numConstraints = 0;
};

class SimulationCUDAContext;

struct SolverData {
    indexType* Tet;
    indexType* Tri;
    glm::vec3* Force;
    glm::vec3* V;
    glm::vec3* X;
    glm::vec3* X0;
    glm::vec3* XTilt;
    glm::vec3* Velocity;
    int numTets;
    int numVerts;
    int numTris;
    glm::vec3* dev_ExtForce;
    glm::mat3* inv_Dm;
};

class Solver {
public:
    Solver(SimulationCUDAContext*, SolverAttribute&);
    virtual ~Solver();

    virtual void Update(SolverData& solverData) = 0;
protected:
    virtual void SolverPrepare(SolverData& solverData) = 0;
    virtual void SolverStep(SolverData& solverData) = 0;
    SolverAttribute attrib;
    const int threadsPerBlock;
    SimulationCUDAContext* mcrpSimContext;

    bool solverReady = false;
};
