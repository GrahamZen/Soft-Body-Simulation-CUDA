#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>

#include <intersections.h>
#include <utilities.h>
#include <glm/gtx/norm.hpp>

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

__host__ __device__ dataType signed_vf_distance(const glmVec3& x,
    const glmVec3& y0, const glmVec3& y1, const glmVec3& y2,
    glmVec3* n, glmVec4& w) {
    *n = cross(normalize(y1 - y0), normalize(y2 - y0));
    if (length2(*n) < 1e-6)
        return FLT_MAX;
    *n = normalize(*n);
    dataType h = dot(x - y0, *n);
    dataType b0 = stp(y1 - x, y2 - x, *n),
        b1 = stp(y2 - x, y0 - x, *n),
        b2 = stp(y0 - x, y1 - x, *n);
    w[0] = 1;
    w[1] = -b0 / (b0 + b1 + b2);
    w[2] = -b1 / (b0 + b1 + b2);
    w[3] = -b2 / (b0 + b1 + b2);
    return h;
}

__host__ __device__ dataType signed_ve_distance(const glmVec3& x, const glmVec3& y0, const glmVec3& y1,
    glmVec3* n, dataType* w) {
    glmVec3 e = y1 - y0;
    dataType d = dot(x - y0, e) / length2(e);
    if (d < 0 || d > 1.0)
        return FLT_MAX;
    if (w) {
        w[0] = 1;
        w[1] = -(1.0 - d);
        w[2] = -d;
        w[3] = 0;
    }
    glmVec3 dist = x - (y0 + d * e);
    dataType l = length(dist);
    if (n && fabs(l) > 1e-16) *n = dist / l;
    return l;
}

__host__ __device__ dataType signed_ee_distance(const glmVec3& x0, const glmVec3& x1,
    const glmVec3& y0, const glmVec3& y1,
    glmVec3* n, glmVec4& w) {
    *n = cross(normalize(x1 - x0), normalize(y1 - y0));
    if (length2(*n) < 1e-6)
        return FLT_MAX;
    *n = normalize(*n);
    dataType h = dot(x0 - y0, *n);
    dataType a0 = stp(y1 - x1, y0 - x1, *n), a1 = stp(y0 - x0, y1 - x0, *n),
        b0 = stp(x0 - y1, x1 - y1, *n), b1 = stp(x1 - y0, x0 - y0, *n);
    w[0] = a0 / (a0 + a1);
    w[1] = a1 / (a0 + a1);
    w[2] = -b0 / (b0 + b1);
    w[3] = -b1 / (b0 + b1);
    return h;
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
            if (t1 > t2) swap(t1, t2);

            tmin = glm::max(tmin, t1);
            tmax = glm::min(tmax, t2);

            if (tmin > tmax) return false;
        }
    }
    return true;
}

__host__ __device__ bool edgeBboxIntersectionTest(const glmVec3& X0, const glmVec3& XTilt, const AABB& bbox, dataType& tmin, dataType& tmax) {
    const dataType eps = glm::epsilon<dataType>();
    glmVec3 d = XTilt - X0;
    glmVec3 ood = 1.0 / d;
    tmin = 0.0;
    tmax = 1.0;

#pragma unroll
    for (int i = 0; i < 3; i++) {
        if (glm::abs<dataType>(d[i]) < eps) {
            if (X0[i] < bbox.min[i] || X0[i] > bbox.max[i]) return false;
        }
        else {
            dataType t1 = (bbox.min[i] - X0[i]) * ood[i];
            dataType t2 = (bbox.max[i] - X0[i]) * ood[i];

            if (t1 > t2) swap(t1, t2);

            tmin = glm::max(tmin, t1);
            tmax = glm::min(tmax, t2);

            if (tmin > tmax) return false;
        }
    }
    return true;
}

__host__ __device__ bool bboxIntersectionTest(const AABB& box1, const AABB& box2) {
    if (box1.max.x < box2.min.x || box1.min.x > box2.max.x) return false;
    if (box1.max.y < box2.min.y || box1.min.y > box2.max.y) return false;
    if (box1.max.z < box2.min.z || box1.min.z > box2.max.z) return false;
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
template dataType solveCubicRange01<dataType>(dataType a, dataType b, dataType c, dataType d, dataType* x);

__host__ __device__ dataType stp(const glmVec3& u, const glmVec3& v, const glmVec3& w) { return glm::dot(u, glm::cross(v, w)); }

__host__ __device__ dataType ccdTriangleIntersectionTest(const glmVec3& x0, const glmVec3& v0,
    const glmVec3& x1, const glmVec3& x2, const glmVec3& x3, const glmVec3& v1, const glmVec3& v2, const glmVec3& v3,
    const glmVec3& xTilt0, const glmVec3& xTilt1, const glmVec3& xTilt2, const glmVec3& xTilt3, glmVec3& n) {
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
    if (abs(a0) < 1e-6 * length(x01) * length(x02) * length(x03))
        return 1.0; // initially coplanar
    dataType t[3];
    dataType minRoot = FLT_MAX;
    int nsol = solveCubic<dataType>(a3, a2, a1, a0, t);
    for (int i = 0; i < nsol; i++) {
        if (t[i] < -1e-3 || t[i] > 1)
            continue;
        glmVec3 xt0 = x0 + t[i] * v0, xt1 = x1 + t[i] * v1,
            xt2 = x2 + t[i] * v2, xt3 = x3 + t[i] * v3;
        glmVec4 w;
        dataType d;
        bool inside;
        d = signed_vf_distance(xt0, xt1, xt2, xt3, &n, w);
        inside = (glm::min(-w[1], glm::min(-w[2], -w[3])) >= -1e-3);
        if (glm::dot(n, w[1] * v1 + w[2] * v2 + w[3] * v3) > 0)
            n = -n;
        if (abs(d) < 1e-6 && inside)
            return t[i];
    }
    return 1.0;
}

__host__ __device__ dataType ccdCollisionTest(const Query& query, const glm::vec3* Xs, const glm::vec3* XTilts, glmVec3& n) {
    const glmVec3 x0 = Xs[query.v0];
    const glmVec3 x1 = Xs[query.v1];
    const glmVec3 x2 = Xs[query.v2];
    const glmVec3 x3 = Xs[query.v3];
    const glmVec3 v0 = glmVec3{ XTilts[query.v0] } - x0;
    const glmVec3 v1 = glmVec3{ XTilts[query.v1] } - x1;
    const glmVec3 v2 = glmVec3{ XTilts[query.v2] } - x2;
    const glmVec3 v3 = glmVec3{ XTilts[query.v3] } - x3;
    const glmVec3 x01 = x1 - x0;
    const glmVec3 x02 = x2 - x0;
    const glmVec3 x03 = x3 - x0;
    const glmVec3 v01 = v1 - v0;
    const glmVec3 v02 = v2 - v0;
    const glmVec3 v03 = v3 - v0;
    const dataType a0 = stp(x01, x02, x03);
    const dataType a1 = stp(v01, x02, x03) + stp(x01, v02, x03) + stp(x01, x02, v03);
    const dataType a2 = stp(x01, v02, v03) + stp(v01, x02, v03) + stp(v01, v02, x03);
    const dataType a3 = stp(v01, v02, v03);
    if (abs(a0) < 1e-6 * length(x01) * length(x02) * length(x03))
        return 1.0; // initially coplanar
    dataType t[3];
    dataType minRoot = FLT_MAX;
    int nsol = solveCubic<dataType>(a3, a2, a1, a0, t);
    for (int i = 0; i < nsol; i++) {
        if (t[i] < -1e-3 || t[i] > 1)
            continue;
        glmVec3 xt0 = x0 + t[i] * v0, xt1 = x1 + t[i] * v1,
            xt2 = x2 + t[i] * v2, xt3 = x3 + t[i] * v3;
        glmVec4 w;
        dataType d;
        bool inside;
        if (query.type == QueryType::VF)
        {
            d = signed_vf_distance(xt0, xt1, xt2, xt3, &n, w);
            inside = (glm::min(-w[1], glm::min(-w[2], -w[3])) >= -1e-3);
        }
        else if(query.type == QueryType::EE)
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