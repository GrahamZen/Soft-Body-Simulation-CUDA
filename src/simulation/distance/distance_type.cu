#include <distance/distance_type.h>
#include <glm/gtc/matrix_inverse.hpp> 
#include <glm/gtx/norm.hpp> 

using namespace ipc;

template <typename Scalar>
__global__ void GetDistanceType(const glm::tvec3<Scalar>* Xs, Query* queries, int numQueries) {
    int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qIdx >= numQueries) return;
    Query& q = queries[qIdx];
    glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
    if (q.type == QueryType::EE) {
        q.dType = edge_edge_distance_type(x0, x1, x2, x3);
    }
    else if (q.type == QueryType::VF) {
        q.dType = point_triangle_distance_type(x0, x1, x2, x3);
    }
}

template __global__ void GetDistanceType<float>(const glm::tvec3<float>* Xs, Query* queries, int numQueries);
template __global__ void GetDistanceType<double>(const glm::tvec3<double>* Xs, Query* queries, int numQueries);

template <typename Scalar>
__global__ void ComputeDistance(const glm::tvec3<Scalar>* Xs, Query* queries, int numQueries, Scalar dhat) {
    int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qIdx >= numQueries) return;
    Query& q = queries[qIdx];
    glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
    if (q.type == QueryType::EE) {
        q.d = edge_edge_distance(x0, x1, x2, x3, q.dType);
    }
    else if (q.type == QueryType::VF) {
        q.d = point_triangle_distance(x0, x1, x2, x3, q.dType);
    }
    //if (q.d > dhat)
    //    q.type = QueryType::UNKNOWN;
}

template __global__ void ComputeDistance<float>(const glm::tvec3<float>* Xs, Query* queries, int numQueries, float dhat);
template __global__ void ComputeDistance<double>(const glm::tvec3<double>* Xs, Query* queries, int numQueries, double dhat);

/// @brief Solve the least square problem: min ||A * x - b||^2
/// @note A = [t1 - t0, glm::cross(t1 - t0, normal)], b = p - t0
/// @param p The point to project
/// @param t0 One end of the edge
/// @param t1 The other end of the edge
/// @param normal The normal of the triangle
/// @return The projected coordinate of the point w.r.t. the coordinate system defined by the edge and normal
template<typename Scalar>
__forceinline__ __device__ glm::tvec2<Scalar> computeProjectedCoordinate(
    const glm::tvec3<Scalar>& p,
    const glm::tvec3<Scalar>& t0,
    const glm::tvec3<Scalar>& t1,
    const glm::tvec3<Scalar>& normal) {
    glm::tmat2x3<Scalar> basis;
    basis[0] = t1 - t0;
    basis[1] = glm::cross(basis[0], normal);
    glm::tmat2x2<Scalar> basisT_basis = glm::tmat2x2<Scalar>(
        glm::dot(basis[0], basis[0]), glm::dot(basis[0], basis[1]),
        glm::dot(basis[1], basis[0]), glm::dot(basis[1], basis[1])
    );
    return glm::inverse(basisT_basis) * glm::tvec2<Scalar>(
        glm::dot(basis[0], p - t0), glm::dot(basis[1], p - t0)
    );
}

template<typename Scalar>
__device__ DistanceType point_triangle_distance_type(
    const glm::tvec3<Scalar>& p,
    const glm::tvec3<Scalar>& t0,
    const glm::tvec3<Scalar>& t1,
    const glm::tvec3<Scalar>& t2)
{
    glm::tvec3<Scalar> normal = glm::cross(t1 - t0, t2 - t0);
    glm::tmat3x2<Scalar> param;

    param[0] = computeProjectedCoordinate(p, t0, t1, normal);
    if (param[0][0] > 0.0 && param[0][0] < 1.0 && param[0][1] >= 0.0) {
        return DistanceType::P_E0;
    }

    param[1] = computeProjectedCoordinate(p, t1, t2, normal);
    if (param[1][0] > 0.0 && param[1][0] < 1.0 && param[1][1] >= 0.0) {
        return DistanceType::P_E1;
    }

    param[2] = computeProjectedCoordinate(p, t2, t0, normal);
    if (param[2][0] > 0.0 && param[2][0] < 1.0 && param[2][1] >= 0.0) {
        return DistanceType::P_E2;
    }

    if (param[0][0] <= 0.0 && param[2][0] >= 1.0) {
        return DistanceType::P_T0;
    }
    else if (param[1][0] <= 0.0 && param[0][0] >= 1.0) {
        return DistanceType::P_T1;
    }
    else if (param[2][0] <= 0.0 && param[1][0] >= 1.0) {
        return DistanceType::P_T2;
    }

    return DistanceType::P_T;
}

template __device__ DistanceType point_triangle_distance_type<float>(
    const glm::tvec3<float>& p,
    const glm::tvec3<float>& t0,
    const glm::tvec3<float>& t1,
    const glm::tvec3<float>& t2);

template __device__ DistanceType point_triangle_distance_type<double>(
    const glm::tvec3<double>& p,
    const glm::tvec3<double>& t0,
    const glm::tvec3<double>& t1,
    const glm::tvec3<double>& t2);

template<typename Scalar>
__device__ DistanceType edge_edge_distance_type(
    const glm::tvec3<Scalar>& ea0,
    const glm::tvec3<Scalar>& ea1,
    const glm::tvec3<Scalar>& eb0,
    const glm::tvec3<Scalar>& eb1)
{
    const Scalar PARALLEL_THRESHOLD = static_cast<Scalar>(1.0e-20);

    const glm::tvec3<Scalar> u = ea1 - ea0;
    const glm::tvec3<Scalar> v = eb1 - eb0;
    const glm::tvec3<Scalar> w = ea0 - eb0;

    const Scalar a = glm::dot(u, u);
    const Scalar b = glm::dot(u, v);
    const Scalar c = glm::dot(v, v);
    const Scalar d = glm::dot(u, w);
    const Scalar e = glm::dot(v, w);
    const Scalar D = a * c - b * b;

    // Degenerate cases: when both edges are points or either edge is degenerate
    if (a == 0.0 && c == 0.0) {
        return DistanceType::EA0_EB0; // Both edges are degenerate (points)
    }
    else if (a == 0.0) {
        return DistanceType::EA0_EB;  // Edge A is degenerate (point), check Edge B
    }
    else if (c == 0.0) {
        return DistanceType::EA_EB0;  // Edge B is degenerate (point), check Edge A
    }

    const Scalar parallel_tolerance = PARALLEL_THRESHOLD * glm::max(static_cast<Scalar>(1.0), a * c);
    if (glm::length2(glm::cross(u, v)) < parallel_tolerance) {
        return edge_edge_parallel_distance_type(ea0, ea1, eb0, eb1);
    }

    DistanceType default_case = DistanceType::EA_EB;

    // Compute the line parameters of the two closest points
    Scalar sN = (b * e - c * d); // sc numerator
    Scalar tN, tD; // tc = tN / tD
    if (sN <= 0.0) { // sc < 0 ⟹ the s=0 edge is visible
        tN = e;
        tD = c;
        default_case = DistanceType::EA0_EB;
    }
    else if (sN >= D) { // sc > 1 ⟹ the s=1 edge is visible
        tN = e + b;
        tD = c;
        default_case = DistanceType::EA1_EB;
    }
    else {
        tN = (a * e - b * d);
        tD = D; // default tD = D ≥ 0

        if (tN > 0.0 && tN < tD && glm::length2(glm::cross(u, v)) < parallel_tolerance) {
            // Avoid coplanar or nearly parallel EE case
            if (sN < D / 2) {
                tN = e;
                tD = c;
                default_case = DistanceType::EA0_EB;
            }
            else {
                tN = e + b;
                tD = c;
                default_case = DistanceType::EA1_EB;
            }
        }
        // Else, default_case stays EA_EB
    }

    // Handling for tc < 0 and tc > 1 cases
    if (tN <= 0.0) { // tc < 0 ⟹ the t=0 edge is visible
        if (-d <= 0.0) {
            return DistanceType::EA0_EB0;
        }
        else if (-d >= a) {
            return DistanceType::EA1_EB0;
        }
        else {
            return DistanceType::EA_EB0;
        }
    }
    else if (tN >= tD) { // tc > 1 ⟹ the t=1 edge is visible
        if ((-d + b) <= 0.0) {
            return DistanceType::EA0_EB1;
        }
        else if ((-d + b) >= a) {
            return DistanceType::EA1_EB1;
        }
        else {
            return DistanceType::EA_EB1;
        }
    }

    return default_case;
}

template<typename Scalar>
__device__ DistanceType edge_edge_parallel_distance_type(
    const glm::tvec3<Scalar>& ea0,
    const glm::tvec3<Scalar>& ea1,
    const glm::tvec3<Scalar>& eb0,
    const glm::tvec3<Scalar>& eb1)
{
    const glm::tvec3<Scalar> ea = ea1 - ea0;
    const double alpha = glm::dot(eb0 - ea0, ea) / glm::length2(ea);
    const double beta = glm::dot(eb1 - ea0, ea) / glm::length2(ea);

    uint8_t eac; // 0: EA0, 1: EA1, 2: EA
    uint8_t ebc; // 0: EB0, 1: EB1, 2: EB
    if (alpha < 0) {
        eac = (0 <= beta && beta <= 1) ? 2 : 0;
        ebc = (beta <= alpha) ? 0 : (beta <= 1 ? 1 : 2);
    }
    else if (alpha > 1) {
        eac = (0 <= beta && beta <= 1) ? 2 : 1;
        ebc = (beta >= alpha) ? 0 : (0 <= beta ? 1 : 2);
    }
    else {
        eac = 2;
        ebc = 0;
    }

    // f(0, 0) = 0000 = 0 -> EA0_EB0
    // f(0, 1) = 0001 = 1 -> EA0_EB1
    // f(1, 0) = 0010 = 2 -> EA1_EB0
    // f(1, 1) = 0011 = 3 -> EA1_EB1
    // f(2, 0) = 0100 = 4 -> EA_EB0
    // f(2, 1) = 0101 = 5 -> EA_EB1
    // f(0, 2) = 0110 = 6 -> EA0_EB
    // f(1, 2) = 0111 = 7 -> EA1_EB
    // f(2, 2) = 1000 = 8 -> EA_EB

    assert(eac != 2 || ebc != 2); // This case results in a degenerate line-line
    return DistanceType((ebc < 2 ? (eac << 1 | ebc) : (6 + eac)) + 7);
}

template __device__ DistanceType edge_edge_distance_type<float>(
    const glm::tvec3<float>& ea0,
    const glm::tvec3<float>& ea1,
    const glm::tvec3<float>& eb0,
    const glm::tvec3<float>& eb1);

template __device__ DistanceType edge_edge_distance_type<double>(
    const glm::tvec3<double>& ea0,
    const glm::tvec3<double>& ea1,
    const glm::tvec3<double>& eb0,
    const glm::tvec3<double>& eb1);