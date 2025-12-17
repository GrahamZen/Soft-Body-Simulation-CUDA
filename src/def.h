#pragma once
#include <glm/glm.hpp>

using indexType = unsigned int;
enum class Precision { 
    Float32, 
    Float64 
};
class FixedBodyData;

template<typename Scalar>
class CollisionDetection;
class Query;

struct MouseSelection {
    int select_v = -1;
    bool dragging = false;
    glm::vec3 target;
    glm::vec3 dir;
};

template<typename Scalar>
struct SolverData {
    indexType* Tri = nullptr;
    indexType* Tet = nullptr;
    glm::tvec3<Scalar>* Force = nullptr;
    glm::tvec3<Scalar>* V = nullptr;
    glm::tvec3<Scalar>* X = nullptr;
    glm::tvec3<Scalar>* X0 = nullptr;
    glm::tvec3<Scalar>* XTilde = nullptr;
    glm::tvec3<Scalar>* DBCX = nullptr;
    Scalar* DBC = nullptr;
    Scalar* moreDBC = nullptr;
    glm::tvec3<Scalar>* OffsetX = nullptr;
    indexType* DBCIdx = nullptr;
    Scalar* mass = nullptr;
    Scalar* mu = nullptr;
    Scalar* lambda = nullptr;
    Scalar* contact_area = nullptr;
    glm::tvec3<Scalar>* ExtForce = nullptr;
    glm::tmat3x3<Scalar>* DmInv = nullptr;
    Scalar* dev_tIs = nullptr;
    glm::vec3* dev_Normals = nullptr;
    FixedBodyData* pFixedBodies = nullptr;
    Scalar* V0 = nullptr;
    indexType* dev_TriFathers;
    indexType* dev_Edges;
    int numDBC = 0;
    int numTris = 0;
    int numTets = 0;
    int numVerts = 0;
    int numQueries() const;
    Query* queries() const;
    CollisionDetection<Scalar>* pCollisionDetection = nullptr;
    MouseSelection mouseSelection;
    Scalar kappa = 1e5;
};

struct SoftBodyAttribute {
    float mass = 1.0f;
    float mu = 20000.0f;
    float lambda = 5000.0f;
    indexType* DBC = nullptr;
    size_t numDBC = 0;
    bool jump = false;
};

struct SolverParamsUI {
    SoftBodyAttribute softBodyAttr;
    size_t numIterations = 1;
    size_t maxIterations = 100;
    double dt = 0.001;
    double damp = 0.999;
    double muN = 0.5;
    double muT = 0.5;
    double gravity = 9.8;
    double dhat = 1e-2;
    double tol = 1e-2;
    double rho = 0.9992;
    bool handleCollision = true;
};

template<typename Scalar>
struct SolverParams {
    SoftBodyAttribute softBodyAttr;
    size_t numIterations = 1;
    size_t maxIterations = 100;
    Scalar dt = 0.001f;
    Scalar damp = 0.999f;
    Scalar muN = 0.5f;
    Scalar muT = 0.5f;
    Scalar gravity = 9.8f;
    // IPC
    Scalar dhat = 1e-2;
    Scalar tol = 1e-2;

    // Pd(Jacobi)
    Scalar rho = 0.9992;

    bool handleCollision = true;
};