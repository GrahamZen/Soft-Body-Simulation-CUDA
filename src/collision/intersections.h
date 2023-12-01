#pragma once

#include <glm/glm.hpp>
#include <sceneStructs.h>
#include <bvh.h>
#include <cuda_runtime.h>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a);
// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t);
/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v);
__host__ __device__ bool edgeBboxIntersectionTest(const glm::vec3& X0, const glm::vec3& XTilt, const AABB& bbox);
__host__ __device__ float tetrahedronTrajIntersectionTest(const glm::vec3& X0, const glm::vec3& XTilt, const glm::vec3* Xs, const glm::vec3* XTilts, GLuint tetId);