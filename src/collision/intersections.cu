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

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glmVec3 multiplyMV(glmMat4 m, glmVec4 v) {
    return glmVec3(m * v);
}

__host__ __device__ bool edgeBboxIntersectionTest(const glmVec3& X0, const glmVec3& XTilt, const AABB& bbox) {
    const dataType eps = glm::epsilon<dataType>();
    glmVec3 d = XTilt - X0;
    glmVec3 ood = 1.0 / d;
    dataType tmin = 0.0;
    dataType tmax = 1.0;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        if (glm::abs<dataType>(d[i]) < eps) {
            if (X0[i] < bbox.min[i] || X0[i] > bbox.max[i]) return false;
        }
        else {
            dataType t1 = (bbox.min[i] - X0[i]) * ood[i];
            dataType t2 = (bbox.max[i] - X0[i]) * ood[i];
            if (t1 > t2) std::swap(t1, t2);

            tmin = glm::max(tmin, t1);
            tmax = glm::min(tmax, t2);

            if (tmin > tmax) return false;
        }
    }
    return true;
}

template<typename T>
__host__ __device__ int solveQuadratic(T a, T b, T c, T* x) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#dataTypeing_point_implementation
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

__host__ __device__ dataType solveCubicMinGtZero(dataType a, dataType b, dataType c, dataType d, dataType defaultVal) {
    double roots[3];
    dataType minRoot = FLT_MAX;
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

__host__ __device__ dataType stp(const glmVec3& u, const glmVec3& v, const glmVec3& w) { return glm::dot(u, glm::cross(v, w)); }
__host__ __device__ dataType norm(const glmVec3& x) {
    return glm::sqrt(glm::dot(x, x));
}
__host__ __device__ dataType ccdTriangleIntersectionTest(const glmVec3& x0, const glmVec3& v0,
    const glmVec3& x1, const glmVec3& x2, const glmVec3& x3, const glmVec3& v1, const glmVec3& v2, const glmVec3& v3) {
    glmVec3 x01 = x1 - x0;
    glmVec3 x02 = x2 - x0;
    glmVec3 x03 = x3 - x0;
    glmVec3 v01 = v1 - v0;
    glmVec3 v02 = v2 - v0;
    glmVec3 v03 = v3 - v0;
    dataType a0 = stp(x01, x02, x03);
    dataType a1 = stp(v01, x02, x03) + stp(x01, v02, x03) + stp(x01, x02, v03);
    dataType a2 = stp(x01, v02, v03) + stp(v01, x02, v03) + stp(v01, v02, x03);
    dataType a3 = stp(v01, v02, v03);
    if (abs(a0) < 1e-6 * norm(x01) * norm(x02) * norm(x03))
        return 1.0; // initially coplanar
    return solveCubicMinGtZero(a3, a2, a1, a0);
}

__host__ __device__ dataType tetrahedronTrajIntersectionTest(const GLuint* tets, const glmVec3& X0, const glmVec3& XTilt, const glm::vec3* Xs, const glm::vec3* XTilts, GLuint tetId) {
    const glmVec3& V0 = XTilt - X0;

    const glmVec3& x0 = Xs[tets[tetId * 4 + 0]];
    const glmVec3& x1 = Xs[tets[tetId * 4 + 1]];
    const glmVec3& x2 = Xs[tets[tetId * 4 + 2]];
    const glmVec3& x3 = Xs[tets[tetId * 4 + 3]];

    const glmVec3& xTilt0 = XTilts[tets[tetId * 4 + 0]];
    const glmVec3& xTilt1 = XTilts[tets[tetId * 4 + 1]];
    const glmVec3& xTilt2 = XTilts[tets[tetId * 4 + 2]];
    const glmVec3& xTilt3 = XTilts[tets[tetId * 4 + 3]];

    const glmVec3 v0 = xTilt0 - x0;
    const glmVec3 v1 = xTilt1 - x1;
    const glmVec3 v2 = xTilt2 - x2;
    const glmVec3 v3 = xTilt3 - x3;
    dataType t = 1.f;
    // pre-test to check if the point is in the same plane as the triangle
        // check if the trajectory intersects with the triangle formed by the first three vertices
    t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x1, x2, v0, v1, v2));
    t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x1, x3, v0, v1, v3));
    t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x2, x3, v0, v2, v3));
    t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x1, x2, x3, v1, v2, v3));
    return t;
}