#pragma once

#include <def.h>

/// @brief Closest pair between a point and triangle.
enum class DistanceType {
    P_T0, ///< The point is closest to triangle vertex zero.
    P_T1, ///< The point is closest to triangle vertex one.
    P_T2, ///< The point is closest to triangle vertex two.
    P_E0, ///< The point is closest to triangle edge zero (vertex zero to one).
    P_E1, ///< The point is closest to triangle edge one (vertex one to two).
    P_E2, ///< The point is closest to triangle edge two (vertex two to zero).
    P_T,  ///< The point is closest to the interior of the triangle.
    /// The edges are closest at vertex 0 of edge A and 0 of edge B.
    EA0_EB0,
    /// The edges are closest at vertex 0 of edge A and 1 of edge B.
    EA0_EB1,
    /// The edges are closest at vertex 1 of edge A and 0 of edge B.
    EA1_EB0,
    /// The edges are closest at vertex 1 of edge A and 1 of edge B.
    EA1_EB1,
    /// The edges are closest at the interior of edge A and vertex 0 of edge B.
    EA_EB0,
    /// The edges are closest at the interior of edge A and vertex 1 of edge B.
    EA_EB1,
    /// The edges are closest at vertex 0 of edge A and the interior of edge B.
    EA0_EB,
    /// The edges are closest at vertex 1 of edge A and the interior of edge B.
    EA1_EB,
    /// The edges are closest at an interior point of edge A and B.
    EA_EB,
    AUTO  ///< Automatically determine the closest pair.
};

extern const char* distanceTypeString[];

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
struct Vec3d {
    double x, y, z;
    Vec3d operator-() const {
        return Vec3d{-x, -y, -z};
    }
};
class Query {
public:
    QueryType type = QueryType::UNKNOWN;
    DistanceType dType = DistanceType::AUTO;
    indexType v0;
    indexType v1;
    indexType v2;
    indexType v3;
    double d;
    double toi = 0.f;
    Vec3d normal = Vec3d{0.0, 0.0, 0.0};
};