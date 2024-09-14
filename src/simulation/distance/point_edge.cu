#include "point_edge.h"

#include <distance/point_line.h>
#include <distance/point_point.h>

#include <stdexcept> // std::invalid_argument

namespace ipc {

    template<typename Scalar>
    __device__ Scalar point_edge_distance(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& e0,
        const glm::tvec3<Scalar>& e1,
        PointEdgeDistanceType dtype) {
        switch (dtype) {
        case PointEdgeDistanceType::P_E0:
            return point_point_distance(p, e0);

        case PointEdgeDistanceType::P_E1:
            return point_point_distance(p, e1);

        case PointEdgeDistanceType::P_E:
            return point_line_distance(p, e0, e1);
        }
    }

    template __device__ float point_edge_distance<float>(
        const glm::tvec3<float>& p,
        const glm::tvec3<float>& e0,
        const glm::tvec3<float>& e1,
        PointEdgeDistanceType dtype);

    template __device__ double point_edge_distance<double>(
        const glm::tvec3<double>& p,
        const glm::tvec3<double>& e0,
        const glm::tvec3<double>& e1,
        PointEdgeDistanceType dtype);

    // template<typename Scalar>
    // VectorMax9d point_edge_distance_gradient(
    //     const glm::tvec3<Scalar>& p,
    //     const glm::tvec3<Scalar>& e0,
    //     const glm::tvec3<Scalar>& e1,
    //     PointEdgeDistanceType dtype)
    // {
    //     const int dim = p.size();
    //     assert(e0.size() == dim);
    //     assert(e1.size() == dim);

    //     if (dtype == PointEdgeDistanceType::AUTO) {
    //         dtype = point_edge_distance_type(p, e0, e1);
    //     }

    //     VectorMax9d grad = VectorMax9d::Zero(3 * dim);

    //     switch (dtype) {
    //     case PointEdgeDistanceType::P_E0:
    //         grad.head(2 * dim) = point_point_distance_gradient(p, e0);
    //         break;

    //     case PointEdgeDistanceType::P_E1: {
    //         const VectorMax6d local_grad = point_point_distance_gradient(p, e1);
    //         grad.head(dim) = local_grad.head(dim);
    //         grad.tail(dim) = local_grad.tail(dim);
    //         break;
    //     }

    //     case PointEdgeDistanceType::P_E:
    //         grad = point_line_distance_gradient(p, e0, e1);
    //         break;

    //     default:
    //         throw std::invalid_argument(
    //             "Invalid distance type for point-edge distance gradient!");
    //     }

    //     return grad;
    // }

    // template<typename Scalar>
    // MatrixMax9d point_edge_distance_hessian(
    //     const glm::tvec3<Scalar>& p,
    //     const glm::tvec3<Scalar>& e0,
    //     const glm::tvec3<Scalar>& e1,
    //     PointEdgeDistanceType dtype)
    // {
    //     const int dim = p.size();
    //     assert(e0.size() == dim);
    //     assert(e1.size() == dim);

    //     if (dtype == PointEdgeDistanceType::AUTO) {
    //         dtype = point_edge_distance_type(p, e0, e1);
    //     }

    //     MatrixMax9d hess = MatrixMax9d::Zero(3 * dim, 3 * dim);

    //     switch (dtype) {
    //     case PointEdgeDistanceType::P_E0:
    //         hess.topLeftCorner(2 * dim, 2 * dim) =
    //             point_point_distance_hessian(p, e0);
    //         break;

    //     case PointEdgeDistanceType::P_E1: {
    //         const MatrixMax6d local_hess = point_point_distance_hessian(p, e1);
    //         hess.topLeftCorner(dim, dim) = local_hess.topLeftCorner(dim, dim);
    //         hess.topRightCorner(dim, dim) = local_hess.topRightCorner(dim, dim);
    //         hess.bottomLeftCorner(dim, dim) = local_hess.bottomLeftCorner(dim, dim);
    //         hess.bottomRightCorner(dim, dim) =
    //             local_hess.bottomRightCorner(dim, dim);
    //         break;
    //     }

    //     case PointEdgeDistanceType::P_E:
    //         hess = point_line_distance_hessian(p, e0, e1);
    //         break;

    //     default:
    //         throw std::invalid_argument(
    //             "Invalid distance type for point-edge distance hessian!");
    //     }

    //     return hess;
    // }

} // namespace ipc
