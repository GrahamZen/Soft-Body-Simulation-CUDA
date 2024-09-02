#pragma once

#include <collision/aabb.h>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a);
/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glmVec3 multiplyMV(glmMat4 m, glmVec4 v);

template<typename T>
inline __host__ __device__ void swap(T& lhs, T& rhs) {
    T tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

template<typename T>
inline __host__ __device__ void sortTwo(T& a, T& b) {
    if (a > b) swap(a, b);
}

template<typename T>
__host__ __device__ void sortThree(T& a, T& b, T& c);

template<typename T>
__host__ __device__ void sortFour(T& a, T& b, T& c, T& d);

__host__ __device__ bool edgeBboxIntersectionTest(const glmVec3& X0, const glmVec3& XTilde, const AABB& bbox);
__host__ __device__ bool edgeBboxIntersectionTest(const glmVec3& X0, const glmVec3& XTilde, const AABB& bbox, colliPrecision& tmin, colliPrecision& tmax);
__host__ __device__ bool bboxIntersectionTest(const AABB& box1, const AABB& box2);
template<typename HighP>
__host__ __device__ colliPrecision ccdCollisionTest(const Query& query, const glm::tvec3<HighP>* Xs, const glm::tvec3<HighP>* XTildes, glmVec3& n);

__host__ __device__ colliPrecision ccdTriangleIntersectionTest(const glmVec3& x0, const glmVec3& v0,
    const glmVec3& x1, const glmVec3& x2, const glmVec3& x3, const glmVec3& v1, const glmVec3& v2, const glmVec3& v3,
    const glmVec3& xTilde0, const glmVec3& xTilde1, const glmVec3& xTilde2, const glmVec3& xTilde3, glmVec3& n);