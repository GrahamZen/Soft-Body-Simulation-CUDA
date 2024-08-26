#pragma once
#include <glm/glm.hpp>

using indexType = unsigned int;
using colliPrecision = double;
using glmVec4 = glm::tvec4<colliPrecision>;
using glmVec3 = glm::tvec3<colliPrecision>;
using glmVec2 = glm::tvec2<colliPrecision>;
using glmMat4 = glm::tmat4x4<colliPrecision>;
using glmMat3 = glm::tmat3x3<colliPrecision>;
using glmMat2 = glm::tmat2x2<colliPrecision>;

class FixedBodyData;

template<typename HighP>
struct SolverData {
    indexType* Tet = nullptr;
    glm::tvec3<HighP>* Force = nullptr;
    glm::tvec3<HighP>* V = nullptr;
    glm::tvec3<HighP>* X = nullptr;
    glm::tvec3<HighP>* X0 = nullptr;
    glm::tvec3<HighP>* XTilde = nullptr;
    HighP* mass = nullptr;
    glm::tvec3<HighP>* dev_ExtForce = nullptr;
    glm::tmat3x3<HighP>* inv_Dm = nullptr;
    colliPrecision* dev_tIs = nullptr;
    glm::vec3* dev_Normals = nullptr;
    FixedBodyData* pFixedBodies = nullptr;
    HighP* V0 = nullptr;
    int numTets = 0;
    int numVerts = 0;
};
