#pragma once

#include <def.h>

template<typename Scalar>
class AABB {
public:
    glm::tvec3<Scalar> min = glm::tvec3<Scalar>{ FLT_MAX };
    glm::tvec3<Scalar> max = glm::tvec3<Scalar>{ -FLT_MAX };
    AABB<Scalar> expand(const AABB<Scalar>& aabb)const;
};

template<typename Scalar>
class BVHNode {
public:
    AABB<Scalar> bbox;
    int isLeaf;
    int leftIndex;
    int rightIndex;
    int parent;
    int TriangleIndex;
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
