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

__host__ __device__ bool edgeTrajBboxIntersectionTest(const glm::vec3& X0, const glm::vec3& XTilt, const glm::vec3& X1, const glm::vec3& X1Tilt, const AABB& bbox) {
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


__host__ __device__ float solveCubic(float a, float b, float c, float d) {
    return 0.f;
}

__host__ __device__ bool ccdTriangleIntersectionPreTest(const glm::vec3& x0, const glm::vec3& xTilt0,
    const glm::vec3& x1, const glm::vec3& xTilt1,
    const glm::vec3& x2, const glm::vec3& xTilt2,
    const glm::vec3& x3, const glm::vec3& xTilt3,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
{
    glm::vec3 n0 = glm::cross(x2 - x1, x3 - x1);
    glm::vec3 n1 = glm::cross(xTilt2 - xTilt1, xTilt3 - xTilt1);
    glm::vec3 term = glm::cross(v2 - v1, v3 - v1);
    glm::vec3 nhat = (n0 + n1 - term) * 0.5f;
    float A = glm::dot(x0 - x1, n0);
    float B = glm::dot(xTilt0 - xTilt1, n1);
    float C = glm::dot(x0 - x1, nhat);
    float D = glm::dot(xTilt0 - xTilt1, nhat);
    float E = glm::dot(x0 - x1, n1);
    float F = glm::dot(xTilt0 - xTilt1, n0);
    bool NonCoplanar = false;
    if (A * B > 0.0f && A * (2.0f * C + F) > 0.0f && A * (2.0f * D + E) > 0.0f) {
        NonCoplanar = true;
    }
    return NonCoplanar;
}

// triangle intersection test
// x1, x2, x3 are the vertices of the triangle, x0 is the point
// v1, v2, v3 are the velocities of the vertices of the triangle, v0 is the velocity of the point
// return the time of intersection, FLT_MAX if no intersection
__host__ __device__ float ccdTriangleIntersectionTest(const glm::vec3& x0, const glm::vec3& v0,
    const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3) {
    const glm::vec3 x01 = x0 - x1;
    const glm::vec3 x21 = x2 - x1;
    const glm::vec3 x31 = x3 - x1;
    const glm::vec3 v01 = v0 - v1;
    const glm::vec3 v21 = v2 - v1;
    const glm::vec3 v31 = v3 - v1;
    float a = glm::dot(v01, glm::cross(v21, v31));
    float b = glm::dot(x01, glm::cross(v21, v31)) + glm::dot(v01, glm::cross(v21, x31)) + glm::dot(v01, glm::cross(x21, v31));
    float c = glm::dot(x01, glm::cross(v21, x31)) + glm::dot(x01, glm::cross(x21, v31)) + glm::dot(v01, glm::cross(x21, x31));
    float d = glm::dot(x01, glm::cross(x21, x31));
    float t = solveCubic(a, b, c, d);
    if (t < 0.0f || t > 1.0f) {
        return FLT_MAX;
    }
    return t;
}

__host__ __device__ float tetrahedronTrajIntersectionTest(const glm::vec3& X0, const glm::vec3& XTilt, const glm::vec3* Xs, const glm::vec3* XTilts, GLuint tetId) {
    const glm::vec3& V0 = XTilt - X0;

    const glm::vec3& x0 = Xs[tetId * 4 + 0];
    const glm::vec3& x1 = Xs[tetId * 4 + 1];
    const glm::vec3& x2 = Xs[tetId * 4 + 2];
    const glm::vec3& x3 = Xs[tetId * 4 + 3];

    const glm::vec3& xTilt0 = XTilts[tetId * 4 + 0];
    const glm::vec3& xTilt1 = XTilts[tetId * 4 + 1];
    const glm::vec3& xTilt2 = XTilts[tetId * 4 + 2];
    const glm::vec3& xTilt3 = XTilts[tetId * 4 + 3];

    const glm::vec3 v0 = xTilt0 - x0;
    const glm::vec3 v1 = xTilt1 - x1;
    const glm::vec3 v2 = xTilt2 - x2;
    const glm::vec3 v3 = xTilt3 - x3;

    // check if the current point is one of the vertices of the tetrahedron, if so, there is no intersection
    // check both the original and the tilted vertices, use epsilon to avoid floating point error
    if ((glm::all(glm::equal(X0, x0)) && glm::all(glm::equal(XTilt, xTilt0))) ||
        glm::all(glm::equal(X0, x1)) && glm::all(glm::equal(XTilt, xTilt1)) ||
        glm::all(glm::equal(X0, x2)) && glm::all(glm::equal(XTilt, xTilt2)) ||
        glm::all(glm::equal(X0, x3)) && glm::all(glm::equal(XTilt, xTilt3))) {
        return -1.0f;
    }
    float t = FLT_MAX;
    // pre-test to check if the point is in the same plane as the triangle
    if (ccdTriangleIntersectionPreTest(x0, xTilt0, x1, xTilt1, x2, xTilt2, x3, xTilt3, v0, v1, v2, v3)) {
        // check if the trajectory intersects with the triangle formed by the first three vertices
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x1, x2, v0, v1, v2));
    }
    if (ccdTriangleIntersectionPreTest(x0, xTilt0, x1, xTilt1, x3, xTilt3, x2, xTilt2, v0, v1, v3, v2)) {
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x1, x3, v0, v1, v3));
    }
    if (ccdTriangleIntersectionPreTest(x0, xTilt0, x2, xTilt2, x3, xTilt3, x1, xTilt1, v0, v2, v3, v1)) {
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x2, x3, v0, v2, v3));
    }
    if (ccdTriangleIntersectionPreTest(x1, xTilt1, x2, xTilt2, x3, xTilt3, x0, xTilt0, v1, v2, v3, v0)) {
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x1, x2, x3, v1, v2, v3));
    }
    return t;
}