#pragma once

#include <collision/aabb.h>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a);
/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
template<typename HighP>
__host__ __device__ glm::tvec3<HighP> multiplyMV(glm::tmat4x4<HighP> m, glm::tvec4<HighP> v);

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

template<typename HighP>
__host__ __device__ bool edgeBboxIntersectionTest(const glm::tvec3<HighP>& X0, const glm::tvec3<HighP>& XTilde, const AABB<HighP>& bbox);
template<typename HighP>
__host__ __device__ bool edgeBboxIntersectionTest(const glm::tvec3<HighP>& X0, const glm::tvec3<HighP>& XTilde, const AABB<HighP>& bbox, HighP& tmin, HighP& tmax);
template<typename HighP>
__host__ __device__ bool bboxIntersectionTest(const AABB<HighP>& box1, const AABB<HighP>& box2);
template<typename HighP>
__host__ __device__ HighP ccdCollisionTest(const Query& query, const glm::tvec3<HighP>* Xs, const glm::tvec3<HighP>* XTildes, glm::tvec3<HighP>& n);

template<typename HighP>
__host__ __device__ HighP ccdTriangleIntersectionTest(const glm::tvec3<HighP>& x0, const glm::tvec3<HighP>& v0,
    const glm::tvec3<HighP>& x1, const glm::tvec3<HighP>& x2, const glm::tvec3<HighP>& x3, const glm::tvec3<HighP>& v1, const glm::tvec3<HighP>& v2, const glm::tvec3<HighP>& v3,
    const glm::tvec3<HighP>& xTilde0, const glm::tvec3<HighP>& xTilde1, const glm::tvec3<HighP>& xTilde2, const glm::tvec3<HighP>& xTilde3, glm::tvec3<HighP>& n);