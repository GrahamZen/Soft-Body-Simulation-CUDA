#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include <intersections.h>
#include <utilities.h>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ bool edgeBboxIntersectionTest(const glm::vec3& X0, const glm::vec3& XTilt, const AABB& bbox) {
    glm::vec3 dir = XTilt - X0;
    glm::vec3 invDir = 1.0f / dir;
    glm::vec3 t0s = (bbox.min - X0) * invDir;
    glm::vec3 t1s = (bbox.max - X0) * invDir;
    glm::vec3 tmin = glm::min(t0s, t1s);
    glm::vec3 tmax = glm::max(t0s, t1s);

    float tminMax = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float tmaxMin = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

    bool inside =
        glm::all(glm::greaterThanEqual(X0, bbox.min)) && glm::all(glm::lessThanEqual(X0, bbox.max)) &&
        glm::all(glm::greaterThanEqual(XTilt, bbox.min)) && glm::all(glm::lessThanEqual(XTilt, bbox.max));

    return (tminMax <= tmaxMin && tmaxMin >= 0.0f) || inside;
}

__host__ __device__ bool isPointInParallelogram(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& aTilt, const glm::vec3& bTilt) {
    glm::vec3 ab = b - a;
    glm::vec3 aaTilt = aTilt - a;
    glm::vec3 ap = p - a;

    float u = glm::dot(glm::cross(ab, ap), glm::cross(ab, aaTilt));
    float v = glm::dot(glm::cross(aaTilt, ap), glm::cross(ab, aaTilt));

    return (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f);
}

__host__ __device__ void lineParallelogramIntersection(const glm::vec3& X0, const glm::vec3& XTilt, const glm::vec3& a, const glm::vec3& b, const glm::vec3& aTilt, const glm::vec3& bTilt, float& minT) {
    glm::vec3 n = glm::cross(b - a, aTilt - a);
    n = glm::normalize(n);

    glm::vec3 dir = XTilt - X0;
    float denom = glm::dot(n, dir);

    if (abs(denom) < 1e-6) {
        return;
    }

    float t = glm::dot(n, a - X0) / denom;
    glm::vec3 p = X0 + t * dir;

    if (t >= 0.0f && t <= 1.0f && isPointInParallelogram(p, a, b, aTilt, bTilt) && t < minT) {
        minT = t;
    }
}

__host__ __device__ float tetrahedronTrajIntersectionTest(const glm::vec3& X0, const glm::vec3& XTilt, const glm::vec3* Xs, const glm::vec3* XTilts, GLuint tetId) {
    const glm::vec3& x0 = Xs[tetId * 4 + 0];
    const glm::vec3& x1 = Xs[tetId * 4 + 1];
    const glm::vec3& x2 = Xs[tetId * 4 + 2];
    const glm::vec3& x3 = Xs[tetId * 4 + 3];

    const glm::vec3& xTilt0 = XTilts[tetId * 4 + 0];
    const glm::vec3& xTilt1 = XTilts[tetId * 4 + 1];
    const glm::vec3& xTilt2 = XTilts[tetId * 4 + 2];
    const glm::vec3& xTilt3 = XTilts[tetId * 4 + 3];

    // check if the current point is one of the vertices of the tetrahedron, if so, there is no intersection
    // check both the original and the tilted vertices, use epsilon to avoid floating point error
    if ((glm::all(glm::equal(X0, x0)) && glm::all(glm::equal(XTilt, xTilt0))) ||
        glm::all(glm::equal(X0, x1)) && glm::all(glm::equal(XTilt, xTilt1)) ||
        glm::all(glm::equal(X0, x2)) && glm::all(glm::equal(XTilt, xTilt2)) ||
        glm::all(glm::equal(X0, x3)) && glm::all(glm::equal(XTilt, xTilt3))) {
        return -1.0f;
    }

    float minT = FLT_MAX;

    lineParallelogramIntersection(X0, XTilt, x0, x1, xTilt0, xTilt1, minT);
    lineParallelogramIntersection(X0, XTilt, x1, x2, xTilt1, xTilt2, minT);
    lineParallelogramIntersection(X0, XTilt, x2, x0, xTilt2, xTilt0, minT);
    lineParallelogramIntersection(X0, XTilt, x0, x3, xTilt0, xTilt3, minT);
    lineParallelogramIntersection(X0, XTilt, x1, x3, xTilt1, xTilt3, minT);
    lineParallelogramIntersection(X0, XTilt, x2, x3, xTilt2, xTilt3, minT);

    return minT == FLT_MAX ? -1.0f : minT;
}