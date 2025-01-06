#pragma once

#include <collision/aabb.h>
#include <sceneStructs.h>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a);
/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
template<typename Scalar>
__host__ __device__ glm::tvec3<Scalar> multiplyMV(glm::tmat4x4<Scalar> m, glm::tvec4<Scalar> v);

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

template<typename Scalar>
__host__ __device__ bool edgeBboxIntersectionTest(const glm::tvec3<Scalar>& X0, const glm::tvec3<Scalar>& XTilde, const AABB<Scalar>& bbox);
template<typename Scalar>
__host__ __device__ bool edgeBboxIntersectionTest(const glm::tvec3<Scalar>& X0, const glm::tvec3<Scalar>& XTilde, const AABB<Scalar>& bbox, Scalar& tmin, Scalar& tmax);
template<typename Scalar>
__host__ __device__ bool bboxIntersectionTest(const AABB<Scalar>& box1, const AABB<Scalar>& box2);
template<typename Scalar>
__host__ __device__ Scalar ccdCollisionTest(const Query& query, const glm::tvec3<Scalar>* Xs, const glm::tvec3<Scalar>* XTildes, glm::tvec3<Scalar>& n);

template<typename Scalar>
__host__ __device__ Scalar ccdTriangleIntersectionTest(const glm::tvec3<Scalar>& x0, const glm::tvec3<Scalar>& v0,
    const glm::tvec3<Scalar>& x1, const glm::tvec3<Scalar>& x2, const glm::tvec3<Scalar>& x3, const glm::tvec3<Scalar>& v1, const glm::tvec3<Scalar>& v2, const glm::tvec3<Scalar>& v3,
    const glm::tvec3<Scalar>& xTilde0, const glm::tvec3<Scalar>& xTilde1, const glm::tvec3<Scalar>& xTilde2, const glm::tvec3<Scalar>& xTilde3, glm::tvec3<Scalar>& n);

__host__ __device__ float rayTriangleIntersection(Ray r, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, bool doubleSided);

template<typename Scalar>
indexType raySimCtxIntersection(Ray r, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X);