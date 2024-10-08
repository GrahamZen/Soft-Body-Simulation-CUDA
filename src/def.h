#pragma once
#include <glm/glm.hpp>

using indexType = unsigned int;

class FixedBodyData;

template<typename Scalar>
struct SolverData {
    indexType* Tet = nullptr;
    glm::tvec3<Scalar>* Force = nullptr;
    glm::tvec3<Scalar>* V = nullptr;
    glm::tvec3<Scalar>* X = nullptr;
    glm::tvec3<Scalar>* X0 = nullptr;
    glm::tvec3<Scalar>* XTilde = nullptr;
    indexType* DBC = nullptr;
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
    indexType* dev_TetFathers;
    indexType* dev_Edges; 
    int numDBC = 0;
    int numTets = 0;
    int numVerts = 0;
};
