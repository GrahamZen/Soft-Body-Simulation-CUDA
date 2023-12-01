#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>

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

template<typename T>
__host__ __device__ int solveQuadratic(T a, T b, T c, T* x) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
    T d = b * b - 4 * a * c;
    if (d < 0) {
        x[0] = -b / (2 * a);
        return 0;
    }
    T q = -(b + glm::sign(b) * sqrt(d)) / 2;
    int i = 0;
    if (abs(a) > 1e-12 * abs(q))
        x[i++] = q / a;
    if (abs(q) > 1e-12 * abs(c))
        x[i++] = c / q;
    if (i == 2 && x[0] > x[1]) {
        T tmp = x[0];
        x[0] = x[1];
        x[1] = tmp;
    }
    return i;
}

template<typename T>
__host__ __device__ int solveCubic(T a, T b, T c, T d, T* x) {
    T xc[2];
    int ncrit = solveQuadratic(3 * a, 2 * b, c, xc);
    if (ncrit == 0) {
        x[0] = newtonsMethod(a, b, c, d, xc[0], 0);
        return 1;
    }
    else if (ncrit == 1) {// cubic is actually quadratic
        return solveQuadratic(b, c, d, x);
    }
    else {
        T yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
                        d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
        int i = 0;
        if (yc[0] * a >= 0)
            x[i++] = newtonsMethod(a, b, c, d, xc[0], -1);
        if (yc[0] * yc[1] <= 0) {
            int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
            x[i++] = newtonsMethod(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
        }
        if (yc[1] * a <= 0)
            x[i++] = newtonsMethod(a, b, c, d, xc[1], 1);
        return i;
    }
}

template<typename T>
__host__ __device__ T newtonsMethod(T a, T b, T c, T d, T x0,
    int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        T y0 = d + x0 * (c + x0 * (b + x0 * a)),
            ddy0 = 2 * b + x0 * (6 * a);
        x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        T y = d + x0 * (c + x0 * (b + x0 * a));
        T dy = c + x0 * (2 * b + x0 * 3 * a);
        if (dy == 0)
            return x0;
        T x1 = x0 - y / dy;
        if (abs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

template<typename T>
__host__ __device__ T solveCubicRange01(T a, T b, T c, T d, T* x) {
    T roots[3];
    int j = 0;
    int numRoots = solveCubic(a, b, c, d, roots);
    for (int i = 0; i < numRoots; i++) {
        if (roots[i] >= 0 && roots[i] <= 1) {
            x[j++] = roots[i];
        }
    }
    return j;
}

__host__ __device__ float solveCubicMinGtZero(float a, float b, float c, float d, float defaultVal = -1.0f) {
    double roots[3];
    float minRoot = FLT_MAX;
    int numRoots = solveCubic<double>((double)a, (double)b, (double)c, (double)d, roots);
    for (int i = 0; i < numRoots; i++) {
        if (roots[i] > 0.0 && roots[i] < 1.0) {
            minRoot = glm::min((double)minRoot, roots[i]);
        }
    }
    if (minRoot == FLT_MAX) {
        return defaultVal;
    }
    return minRoot;
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
    return solveCubicMinGtZero(a, b, c, d);
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