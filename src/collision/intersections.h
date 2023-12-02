#pragma once

#include <glm/glm.hpp>
#include <sceneStructs.h>
#include <bvh.h>
#include <cuda_runtime.h>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a);
/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glmVec3 multiplyMV(glmMat4 m, glmVec4 v);
__host__ __device__ bool edgeBboxIntersectionTest(const glmVec3& X0, const glmVec3& XTilt, const AABB& bbox);
__host__ __device__ bool edgeBboxIntersectionTest(const glmVec3& X0, const glmVec3& XTilt, const AABB& bbox, dataType& tmin, dataType& tmax);
__host__ __device__ bool bboxIntersectionTest(const AABB& box1, const AABB& box2);
template<typename T>
__host__ __device__ int solveQuadratic(T a, T b, T c, T* x);
template<typename T>
__host__ __device__ int solveCubic(T a, T b, T c, T d, T* x);
template<typename T>
__host__ __device__ T solveCubicRange01(T a, T b, T c, T d, T* x);
__host__ __device__ dataType stp(const glmVec3& u, const glmVec3& v, const glmVec3& w);
__host__ __device__ dataType ccdTriangleIntersectionTest(const glmVec3& x0, const glmVec3& v0,
    const glmVec3& x1, const glmVec3& x2, const glmVec3& x3, const glmVec3& v1, const glmVec3& v2, const glmVec3& v3,
    const glmVec3& xTilt0, const glmVec3& xTilt1, const glmVec3& xTilt2, const glmVec3& xTilt3, glmVec3& n);
__host__ __device__ dataType ccdTetrahedronIntersectionTest(const glmVec3& X0, const glmVec3& XTilt,
    const glmVec3& x0, const glmVec3& x1, const glmVec3& x2, const glmVec3& x3,
    const glmVec3 xTilt0, const glmVec3 xTilt1, const glmVec3 xTilt2, const glmVec3 xTilt3, glm::vec3& nor);
__host__ __device__ dataType tetrahedronTrajIntersectionTest(const GLuint* tets, const glmVec3& X0, const glmVec3& XTilt, const glm::vec3* Xs, const glm::vec3* XTilts, GLuint tetId, glm::vec3& nor);