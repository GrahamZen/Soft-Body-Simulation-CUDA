#pragma once

#include <distance/distance_type.h>

namespace ipc {

/// @brief Compute the distance between a point and edge in 2D or 3D.
/// @note The distance is actually squared distance.
/// @param p The point.
/// @param e0 The first vertex of the edge.
/// @param e1 The second vertex of the edge.
/// @param dtype The point edge distance type to compute.
/// @return The distance between the point and edge.
template<typename Scalar>
__host__ __device__ Scalar point_edge_distance(
    const glm::tvec3<Scalar>& p,
    const glm::tvec3<Scalar>& e0,
    const glm::tvec3<Scalar>& e1,
    PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);

/// @brief Compute the gradient of the distance between a point and edge.
/// @note The distance is actually squared distance.
/// @param p The point.
/// @param e0 The first vertex of the edge.
/// @param e1 The second vertex of the edge.
/// @param dtype The point edge distance type to compute.
/// @return grad The gradient of the distance wrt p, e0, and e1.
// template<typename Scalar>
// VectorMax9d point_edge_distance_gradient(
//     const glm::tvec3<Scalar>& p,
//     const glm::tvec3<Scalar>& e0,
//     const glm::tvec3<Scalar>& e1,
//     PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);

// /// @brief Compute the hessian of the distance between a point and edge.
// /// @note The distance is actually squared distance.
// /// @param p The point.
// /// @param e0 The first vertex of the edge.
// /// @param e1 The second vertex of the edge.
// /// @param dtype The point edge distance type to compute.
// /// @return hess The hessian of the distance wrt p, e0, and e1.
// template<typename Scalar>
// MatrixMax9d point_edge_distance_hessian(
//     const glm::tvec3<Scalar>& p,
//     const glm::tvec3<Scalar>& e0,
//     const glm::tvec3<Scalar>& e1,
//     PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);

} // namespace ipc
