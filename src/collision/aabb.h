#pragma once

#include <def.h>

class AABB {
public:
    glm::vec3 min = glm::vec3{ FLT_MAX };
    glm::vec3 max = glm::vec3{ -FLT_MAX };
    AABB expand(const AABB& aabb)const;
};
class BVHNode {
public:
    AABB bbox;
    int isLeaf;
    int leftIndex;
    int rightIndex;
    int parent;
    int TetrahedronIndex;
};

enum class QueryType {
    UNKNOWN,
    VF,
    EE
};

class Query {
public:
    QueryType type;
    indexType v0;
    indexType v1;
    indexType v2;
    indexType v3;
    float toi = 0.f;
    glm::vec3 normal = glm::vec3(0.f);
};
