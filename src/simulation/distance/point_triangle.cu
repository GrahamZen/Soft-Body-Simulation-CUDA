#include "point_triangle.h"

#include <distance/point_line.h>
#include <distance/point_plane.h>
#include <distance/point_point.h>
#include <cuda_runtime.h>

namespace ipc {

    template<typename Scalar>
    __host__ __device__ Scalar point_triangle_distance(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& t0,
        const glm::tvec3<Scalar>& t1,
        const glm::tvec3<Scalar>& t2,
        DistanceType dtype)
    {
        if (dtype == DistanceType::P_T0)
            return point_point_distance(p, t0);

        if (dtype == DistanceType::P_T1)
            return point_point_distance(p, t1);

        if (dtype == DistanceType::P_T2)
            return point_point_distance(p, t2);

        if (dtype == DistanceType::P_E0) {
            return point_line_distance(p, t0, t1);
        }

        if (dtype == DistanceType::P_E1)
            return point_line_distance(p, t1, t2);

        if (dtype == DistanceType::P_E2)
            return point_line_distance(p, t2, t0);

        if (dtype == DistanceType::P_T)
            return point_plane_distance(p, t0, t1, t2);

        else {
            return 0;
        }

    }

    template __host__ __device__ float point_triangle_distance<float>(
        const glm::tvec3<float>& p,
        const glm::tvec3<float>& t0,
        const glm::tvec3<float>& t1,
        const glm::tvec3<float>& t2,
        DistanceType dtype);

    template __host__ __device__ double point_triangle_distance<double>(
        const glm::tvec3<double>& p,
        const glm::tvec3<double>& t0,
        const glm::tvec3<double>& t1,
        const glm::tvec3<double>& t2,
        DistanceType dtype);
    template<typename Scalar>
    __host__ __device__ Vector12<Scalar> point_triangle_distance_gradient(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& t0,
        const glm::tvec3<Scalar>& t1,
        const glm::tvec3<Scalar>& t2,
        DistanceType dtype)
    {
        Vector12<Scalar> grad;

        switch (dtype) {
        case DistanceType::P_T0:
            grad.head(6) = point_point_distance_gradient(p, t0);
            break;

        case DistanceType::P_T1: {
            const Vector<Scalar, 6> local_grad = point_point_distance_gradient(p, t1);
            grad.head(3) = local_grad.head(3);
            grad.segment(3, 6) = local_grad.tail(3);
            break;
        }

        case DistanceType::P_T2: {
            const Vector<Scalar, 6> local_grad = point_point_distance_gradient(p, t2);
            grad.head(3) = local_grad.head(3);
            grad.tail(3) = local_grad.tail(3);
            break;
        }

        case DistanceType::P_E0:
            grad.head(9) = point_line_distance_gradient(p, t0, t1);
            break;

        case DistanceType::P_E1: {
            const Vector<Scalar, 9> local_grad = point_line_distance_gradient(p, t1, t2);
            grad.head(3) = local_grad.head(3);
            grad.tail(6) = local_grad.tail(6);
            break;
        }

        case DistanceType::P_E2: {
            const Vector<Scalar, 9> local_grad = point_line_distance_gradient(p, t2, t0);
            grad.head(3) = local_grad.head(3);     // ∇_p
            grad.segment(3, 3) = local_grad.tail(3); // ∇_{t0}
            grad.tail(3) = local_grad.segment(3, 3); // ∇_{t2}
            break;
        }

        case DistanceType::P_T:
            grad = point_plane_distance_gradient(p, t0, t1, t2);
            break;
        }

        return grad;
    }

    template __host__ __device__ Vector12<float> point_triangle_distance_gradient<float>(
        const glm::tvec3<float>& p,
        const glm::tvec3<float>& t0,
        const glm::tvec3<float>& t1,
        const glm::tvec3<float>& t2,
        DistanceType dtype);

    template __host__ __device__ Vector12<double> point_triangle_distance_gradient<double>(
        const glm::tvec3<double>& p,
        const glm::tvec3<double>& t0,
        const glm::tvec3<double>& t1,
        const glm::tvec3<double>& t2,
        DistanceType dtype);
    template<typename Scalar>
    Matrix12<Scalar> point_triangle_distance_hessian(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& t0,
        const glm::tvec3<Scalar>& t1,
        const glm::tvec3<Scalar>& t2,
        DistanceType dtype)
    {
        Matrix12<Scalar> hess;

        switch (dtype) {
        case DistanceType::P_T0:
            hess.topLeftCorner(6, 6) = point_point_distance_hessian(p, t0);
            break;

        case DistanceType::P_T1: {
            Matrix6<Scalar> local_hess = point_point_distance_hessian(p, t1);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.block(3, 3, 0, 6) = local_hess.topRightCorner(3, 3);
            hess.block(3, 3, 6, 0) = local_hess.bottomLeftCorner(3, 3);
            hess.block(3, 3, 6, 6) = local_hess.bottomRightCorner(3, 3);
            break;
        }

        case DistanceType::P_T2: {
            Matrix6<Scalar> local_hess = point_point_distance_hessian(p, t2);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.topRightCorner(3, 3) = local_hess.topRightCorner(3, 3);
            hess.bottomLeftCorner(3, 3) = local_hess.bottomLeftCorner(3, 3);
            hess.bottomRightCorner(3, 3) = local_hess.bottomRightCorner(3, 3);
            break;
        }

        case DistanceType::P_E0:
            hess.topLeftCorner(9, 9) = point_line_distance_hessian(p, t0, t1);
            break;

        case DistanceType::P_E1: {
            Matrix9<Scalar> local_hess = point_line_distance_hessian(p, t1, t2);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.topRightCorner(3, 6) = local_hess.topRightCorner(3, 6);
            hess.bottomLeftCorner(6, 3) = local_hess.bottomLeftCorner(6, 3);
            hess.bottomRightCorner(6, 6) = local_hess.bottomRightCorner(6, 6);
            break;
        }

        case DistanceType::P_E2: {
            Matrix9<Scalar> local_hess = point_line_distance_hessian(p, t2, t0);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.block(3, 3, 0, 3) = local_hess.topRightCorner(3, 3);
            hess.topRightCorner(3, 3) = local_hess.block(3, 3, 0, 3);
            hess.block(3, 3, 3, 0) = local_hess.bottomLeftCorner(3, 3);
            hess.block(3, 3, 3, 3) = local_hess.bottomRightCorner(3, 3);
            hess.block(3, 3, 3, 9) = local_hess.block(3, 3, 6, 3);
            hess.bottomLeftCorner(3, 3) = local_hess.block(3, 3, 3, 0);
            hess.block(3, 3, 9, 3) = local_hess.block(3, 3, 3, 6);
            hess.bottomRightCorner(3, 3) = local_hess.block(3, 3, 3, 3);
            break;
        }

        case DistanceType::P_T:
            hess = point_plane_distance_hessian(p, t0, t1, t2);
            break;
        }

        return hess;
    }

    template Matrix12<float> point_triangle_distance_hessian<float>(
        const glm::tvec3<float>& p,
        const glm::tvec3<float>& t0,
        const glm::tvec3<float>& t1,
        const glm::tvec3<float>& t2,
        DistanceType dtype);

    template Matrix12<double> point_triangle_distance_hessian<double>(
        const glm::tvec3<double>& p,
        const glm::tvec3<double>& t0,
        const glm::tvec3<double>& t1,
        const glm::tvec3<double>& t2,
        DistanceType dtype);

} // namespace ipc
