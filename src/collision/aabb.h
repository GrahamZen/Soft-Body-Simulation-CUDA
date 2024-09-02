#pragma once

#include <def.h>

template<typename HighP>
class AABB {
public:
    glm::tvec3<HighP> min = glm::tvec3<HighP>{ FLT_MAX };
    glm::tvec3<HighP> max = glm::tvec3<HighP>{ -FLT_MAX };
    AABB<HighP> expand(const AABB<HighP>& aabb)const;
};

template<typename HighP>
class BVHNode {
public:
    AABB<HighP> bbox;
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
