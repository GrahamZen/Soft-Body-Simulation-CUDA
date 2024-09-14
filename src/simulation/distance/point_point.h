#pragma once

#include<glm/glm.hpp>

namespace ipc {

/// @brief Compute the distance between two points.
/// @note The distance is actually squared distance.
/// @param p0 The first point.
/// @param p1 The second point.
/// @return The distance between p0 and p1.
    template<typename Scalar>
    __device__ Scalar point_point_distance(
    const glm::tvec3<Scalar>& p0,
    const glm::tvec3<Scalar>& p1);

/// @brief Compute the gradient of the distance between two points.
/// @note The distance is actually squared distance.
/// @param p0 The first point.
/// @param p1 The second point.
/// @return The computed gradient.
// VectorMax6d point_point_distance_gradient(
//     const glm::tvec3<Scalar>& p0,
//     const glm::tvec3<Scalar>& p1);

// /// @brief Compute the hessian of the distance between two points.
// /// @note The distance is actually squared distance.
// /// @param p0 The first point.
// /// @param p1 The second point.
// /// @return The computed hessian.
// MatrixMax6d point_point_distance_hessian(
//     const glm::tvec3<Scalar>& p0,
//     const glm::tvec3<Scalar>& p1);


} // namespace ipc
