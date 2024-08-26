#pragma once

#include <def.h>
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


template<typename T>
__host__ __device__ T newtonsMethod(T a, T b, T c, T d, T x0, int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        T y0 = d + x0 * (c + x0 * (b + x0 * a)),
            ddy0 = 2 * b + x0 * (6 * a);
        if (ddy0 != 0)
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
__host__ __device__ T solveCubicRange01(T a, T b, T c, T d, T* x);
__host__ __device__ colliPrecision stp(const glmVec3& u, const glmVec3& v, const glmVec3& w);

__host__ __device__ colliPrecision signed_vf_distance(const glmVec3& x,
    const glmVec3& y0, const glmVec3& y1, const glmVec3& y2,
    glmVec3* n, glmVec4& w);

__host__ __device__ colliPrecision signed_ee_distance(const glmVec3& x0, const glmVec3& x1,
    const glmVec3& y0, const glmVec3& y1,
    glmVec3* n, glmVec4& w);

template<typename HighP>
__host__ __device__ colliPrecision ccdCollisionTest(const Query& query, const glm::tvec3<HighP>* Xs, const glm::tvec3<HighP>* XTildes, glmVec3& n) {
    const glmVec3 x0 = Xs[query.v0];
    const glmVec3 x1 = Xs[query.v1];
    const glmVec3 x2 = Xs[query.v2];
    const glmVec3 x3 = Xs[query.v3];
    const glmVec3 v0 = glmVec3{ XTildes[query.v0] } - x0;
    const glmVec3 v1 = glmVec3{ XTildes[query.v1] } - x1;
    const glmVec3 v2 = glmVec3{ XTildes[query.v2] } - x2;
    const glmVec3 v3 = glmVec3{ XTildes[query.v3] } - x3;
    const glmVec3 x01 = x1 - x0;
    const glmVec3 x02 = x2 - x0;
    const glmVec3 x03 = x3 - x0;
    const glmVec3 v01 = v1 - v0;
    const glmVec3 v02 = v2 - v0;
    const glmVec3 v03 = v3 - v0;
    const colliPrecision a0 = stp(x01, x02, x03);
    const colliPrecision a1 = stp(v01, x02, x03) + stp(x01, v02, x03) + stp(x01, x02, v03);
    const colliPrecision a2 = stp(x01, v02, v03) + stp(v01, x02, v03) + stp(v01, v02, x03);
    const colliPrecision a3 = stp(v01, v02, v03);
    if (abs(a0) < 1e-6 * length(x01) * length(x02) * length(x03))
        return 1.0; // initially coplanar
    colliPrecision t[3];
    colliPrecision minRoot = FLT_MAX;
    int nsol = solveCubic<colliPrecision>(a3, a2, a1, a0, t);
    for (int i = 0; i < nsol; i++) {
        if (t[i] < -1e-3 || t[i] > 1)
            continue;
        glmVec3 xt0 = x0 + t[i] * v0, xt1 = x1 + t[i] * v1,
            xt2 = x2 + t[i] * v2, xt3 = x3 + t[i] * v3;
        glmVec4 w;
        colliPrecision d;
        bool inside;
        if (query.type == QueryType::VF)
        {
            d = signed_vf_distance(xt0, xt1, xt2, xt3, &n, w);
            inside = (glm::min(-w[1], glm::min(-w[2], -w[3])) >= -1e-3);
        }
        else if (query.type == QueryType::EE)
        {
            d = signed_ee_distance(xt0, xt1, xt2, xt3, &n, w);
            inside = (glm::min(w[0], glm::min(w[1], glm::min(-w[2], -w[3]))) >= -1e-3);
        }
        if (glm::dot(n, w[1] * v1 + w[2] * v2 + w[3] * v3) > 0)
            n = -n;
        if (abs(d) < 1e-6 && inside)
            return t[i];
    }
    return 1.0;
}

__host__ __device__ colliPrecision ccdTriangleIntersectionTest(const glmVec3& x0, const glmVec3& v0,
    const glmVec3& x1, const glmVec3& x2, const glmVec3& x3, const glmVec3& v1, const glmVec3& v2, const glmVec3& v3,
    const glmVec3& xTilde0, const glmVec3& xTilde1, const glmVec3& xTilde2, const glmVec3& xTilde3, glmVec3& n);