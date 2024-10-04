#include "edge_edge.h"

#include <distance/point_point.h>
#include <distance/point_line.h>
#include <distance/line_line.h>

namespace ipc {

    template<typename Scalar>
    __host__ __device__ Scalar edge_edge_distance(
        const glm::tvec3<Scalar>& ea0,
        const glm::tvec3<Scalar>& ea1,
        const glm::tvec3<Scalar>& eb0,
        const glm::tvec3<Scalar>& eb1,
        DistanceType dtype)
    {
        if (dtype == DistanceType::EA_EB) {
            return line_line_distance(ea0, ea1, eb0, eb1);
        }
        else if (dtype == DistanceType::EA0_EB0) {
            return point_point_distance(ea0, eb0);
        }
        else if (dtype == DistanceType::EA0_EB1) {
            return point_point_distance(ea0, eb1);
        }
        else if (dtype == DistanceType::EA1_EB0) {
            return point_point_distance(ea1, eb0);
        }
        else if (dtype == DistanceType::EA1_EB1) {
            return point_point_distance(ea1, eb1);
        }
        else if (dtype == DistanceType::EA_EB0) {
            return point_line_distance(eb0, ea0, ea1);
        }
        else if (dtype == DistanceType::EA_EB1) {
            return point_line_distance(eb1, ea0, ea1);
        }
        else if (dtype == DistanceType::EA0_EB) {
            return point_line_distance(ea0, eb0, eb1);
        }
        else if (dtype == DistanceType::EA1_EB) {
            return point_line_distance(ea1, eb0, eb1);
        }
        else {
            return 0;
        }


    }

    template __host__ __device__ float edge_edge_distance<float>(
        const glm::tvec3<float>& ea0,
        const glm::tvec3<float>& ea1,
        const glm::tvec3<float>& eb0,
        const glm::tvec3<float>& eb1,
        DistanceType dtype);

    template __host__ __device__ double edge_edge_distance<double>(
        const glm::tvec3<double>& ea0,
        const glm::tvec3<double>& ea1,
        const glm::tvec3<double>& eb0,
        const glm::tvec3<double>& eb1,
        DistanceType dtype);

    template<typename Scalar>
    __host__ __device__ Vector12<Scalar> edge_edge_distance_gradient(
        const glm::tvec3<Scalar>& ea0,
        const glm::tvec3<Scalar>& ea1,
        const glm::tvec3<Scalar>& eb0,
        const glm::tvec3<Scalar>& eb1,
        DistanceType dtype)
    {
        Vector12<Scalar> grad;

        switch (dtype) {
        case DistanceType::EA0_EB0: {
            const Vector<Scalar, 6> local_grad = point_point_distance_gradient(ea0, eb0);
            grad.head(3) = local_grad.head(3);
            grad.segment(3, 6) = local_grad.tail(3);
            break;
        }

        case DistanceType::EA0_EB1: {
            const Vector<Scalar, 6> local_grad = point_point_distance_gradient(ea0, eb1);
            grad.head(3) = local_grad.head(3);
            grad.tail(3) = local_grad.tail(3);
            break;
        }

        case DistanceType::EA1_EB0:
            grad.segment(6, 3) = point_point_distance_gradient(ea1, eb0);
            break;

        case DistanceType::EA1_EB1: {
            const Vector<Scalar, 6> local_grad = point_point_distance_gradient(ea1, eb1);
            grad.segment(3, 3) = local_grad.head(3);
            grad.tail(3) = local_grad.tail(3);
            break;
        }

        case DistanceType::EA_EB0: {
            const Vector<Scalar, 9> local_grad = point_line_distance_gradient(eb0, ea0, ea1);
            grad.head(6) = local_grad.tail(6);
            grad.segment(3, 6) = local_grad.head(3);
            break;
        }

        case DistanceType::EA_EB1: {
            const Vector<Scalar, 9> local_grad = point_line_distance_gradient(eb1, ea0, ea1);
            grad.head(6) = local_grad.tail(6);
            grad.tail(3) = local_grad.head(3);
            break;
        }

        case DistanceType::EA0_EB: {
            const Vector<Scalar, 9> local_grad = point_line_distance_gradient(ea0, eb0, eb1);
            grad.head(3) = local_grad.head(3);
            grad.tail(6) = local_grad.tail(6);
            break;
        }

        case DistanceType::EA1_EB:
            grad.tail(9) = point_line_distance_gradient(ea1, eb0, eb1);
            break;

        case DistanceType::EA_EB:
            grad = line_line_distance_gradient(ea0, ea1, eb0, eb1);
            break;

        }

        return grad;
    }

    template __host__ __device__ Vector12<float> edge_edge_distance_gradient(
        const glm::tvec3<float>& ea0,
        const glm::tvec3<float>& ea1,
        const glm::tvec3<float>& eb0,
        const glm::tvec3<float>& eb1,
        DistanceType dtype);

    template __host__ __device__ Vector12<double> edge_edge_distance_gradient(
        const glm::tvec3<double>& ea0,
        const glm::tvec3<double>& ea1,
        const glm::tvec3<double>& eb0,
        const glm::tvec3<double>& eb1,
        DistanceType dtype);

    template<typename Scalar>
    __host__ __device__ Matrix12<Scalar> edge_edge_distance_hessian(
        const glm::tvec3<Scalar>& ea0,
        const glm::tvec3<Scalar>& ea1,
        const glm::tvec3<Scalar>& eb0,
        const glm::tvec3<Scalar>& eb1,
        DistanceType dtype)
    {
        Matrix12<Scalar> hess;

        switch (dtype) {
        case DistanceType::EA0_EB0: {
            Matrix6<Scalar> local_hess = point_point_distance_hessian(ea0, eb0);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.block(3, 3, 0, 6) = local_hess.topRightCorner(3, 3);
            hess.block(3, 3, 6, 0) = local_hess.bottomLeftCorner(3, 3);
            hess.block(3, 3, 6, 6) = local_hess.bottomRightCorner(3, 3);
            break;
        }

        case DistanceType::EA0_EB1: {
            Matrix6<Scalar> local_hess = point_point_distance_hessian(ea0, eb1);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.topRightCorner(3, 3) = local_hess.topRightCorner(3, 3);
            hess.bottomLeftCorner(3, 3) = local_hess.bottomLeftCorner(3, 3);
            hess.bottomRightCorner(3, 3) = local_hess.bottomRightCorner(3, 3);
            break;
        }

        case DistanceType::EA1_EB0:
            hess.block(6, 6, 3, 3) = point_point_distance_hessian(ea1, eb0);
            break;

        case DistanceType::EA1_EB1: {
            Matrix6<Scalar> local_hess = point_point_distance_hessian(ea1, eb1);
            hess.block(3, 3, 3, 3) = local_hess.topLeftCorner(3, 3);
            hess.block(3, 3, 3, 9) = local_hess.topRightCorner(3, 3);
            hess.block(3, 3, 9, 3) = local_hess.bottomLeftCorner(3, 3);
            hess.bottomRightCorner(3, 3) = local_hess.bottomRightCorner(3, 3);
            break;
        }

        case DistanceType::EA_EB0: {
            Matrix9<Scalar> local_hess = point_line_distance_hessian(eb0, ea0, ea1);
            hess.topLeftCorner(6, 6) = local_hess.bottomRightCorner(6, 6);
            hess.block(3, 6, 6, 0) = local_hess.topRightCorner(3, 6);
            hess.block(6, 3, 0, 6) = local_hess.bottomLeftCorner(6, 3);
            hess.block(3, 3, 6, 6) = local_hess.topLeftCorner(3, 3);
            break;
        }

        case DistanceType::EA_EB1: {
            Matrix9<Scalar> local_hess = point_line_distance_hessian(eb1, ea0, ea1);
            hess.topLeftCorner(6, 6) = local_hess.bottomRightCorner(6, 6);
            hess.topRightCorner(6, 3) = local_hess.bottomLeftCorner(6, 3);
            hess.bottomLeftCorner(3, 6) = local_hess.topRightCorner(3, 6);
            hess.bottomRightCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            break;
        }

        case DistanceType::EA0_EB: {
            Matrix9<Scalar> local_hess = point_line_distance_hessian(ea0, eb0, eb1);
            hess.topLeftCorner(3, 3) = local_hess.topLeftCorner(3, 3);
            hess.topRightCorner(3, 6) = local_hess.topRightCorner(3, 6);
            hess.bottomLeftCorner(6, 3) = local_hess.bottomLeftCorner(6, 3);
            hess.bottomRightCorner(6, 6) = local_hess.bottomRightCorner(6, 6);
            break;
        }

        case DistanceType::EA1_EB:
            hess.bottomRightCorner(9, 9) =
                point_line_distance_hessian(ea1, eb0, eb1);
            break;

        case DistanceType::EA_EB:
            hess = line_line_distance_hessian(ea0, ea1, eb0, eb1);
            break;
        }

        return hess;
    }

    template __host__ __device__  Matrix12<float> edge_edge_distance_hessian<float>(
        const glm::tvec3<float>& ea0,
        const glm::tvec3<float>& ea1,
        const glm::tvec3<float>& eb0,
        const glm::tvec3<float>& eb1,
        DistanceType dtype);

    template __host__ __device__  Matrix12<double> edge_edge_distance_hessian<double>(
        const glm::tvec3<double>& ea0,
        const glm::tvec3<double>& ea1,
        const glm::tvec3<double>& eb0,
        const glm::tvec3<double>& eb1,
        DistanceType dtype);

} // namespace ipc
