#pragma once
#include <glm/glm.hpp>

using indexType = unsigned int;

class FixedBodyData;

template<typename HighP>
struct SolverData {
    indexType* Tet = nullptr;
    glm::tvec3<HighP>* Force = nullptr;
    glm::tvec3<HighP>* V = nullptr;
    glm::tvec3<HighP>* X = nullptr;
    glm::tvec3<HighP>* X0 = nullptr;
    glm::tvec3<HighP>* XTilde = nullptr;
    indexType* DBC = nullptr;
    HighP* mass = nullptr;
    HighP* mu = nullptr;
    HighP* lambda = nullptr;
    HighP* contact_area = nullptr;
    glm::tvec3<HighP>* ExtForce = nullptr;
    glm::tmat3x3<HighP>* DmInv = nullptr;
    HighP* dev_tIs = nullptr;
    glm::vec3* dev_Normals = nullptr;
    FixedBodyData* pFixedBodies = nullptr;
    HighP* V0 = nullptr;
    indexType* dev_TetFathers;
    indexType* dev_Edges; 
    int numDBC = 0;
    int numTets = 0;
    int numVerts = 0;
};
