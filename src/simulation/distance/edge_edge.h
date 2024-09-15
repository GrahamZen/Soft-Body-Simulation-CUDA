#pragma once

#include <vector.h>
#include <distance/distance_type.h>

namespace ipc {

/// @brief Compute the distance between a two lines segments in 3D.
/// @note The distance is actually squared distance.
/// @param ea0 The first vertex of the first edge.
/// @param ea1 The second vertex of the first edge.
/// @param eb0 The first vertex of the second edge.
/// @param eb1 The second vertex of the second edge.
/// @param dtype The point edge distance type to compute.
/// @return The distance between the two edges.
template<typename Scalar>
__device__ Scalar edge_edge_distance(
    const glm::tvec3<Scalar>& ea0,
    const glm::tvec3<Scalar>& ea1,
    const glm::tvec3<Scalar>& eb0,
    const glm::tvec3<Scalar>& eb1,
    DistanceType dtype = DistanceType::AUTO);

 /// @brief Compute the gradient of the distance between a two lines segments.
 /// @note The distance is actually squared distance.
 /// @param ea0 The first vertex of the first edge.
 /// @param ea1 The second vertex of the first edge.
 /// @param eb0 The first vertex of the second edge.
 /// @param eb1 The second vertex of the second edge.
 /// @param dtype The point edge distance type to compute.
 /// @return The gradient of the distance wrt ea0, ea1, eb0, and eb1.
 template<typename Scalar>
 __device__ Vector12<Scalar> edge_edge_distance_gradient(
     const glm::tvec3<Scalar>& ea0,
     const glm::tvec3<Scalar>& ea1,
     const glm::tvec3<Scalar>& eb0,
     const glm::tvec3<Scalar>& eb1,
     DistanceType dtype = DistanceType::AUTO);

// /// @brief Compute the hessian of the distance between a two lines segments.
// /// @note The distance is actually squared distance.
// /// @param ea0 The first vertex of the first edge.
// /// @param ea1 The second vertex of the first edge.
// /// @param eb0 The first vertex of the second edge.
// /// @param eb1 The second vertex of the second edge.
// /// @param dtype The point edge distance type to compute.
// /// @return The hessian of the distance wrt ea0, ea1, eb0, and eb1.
// template<typename Scalar>
// Matrix12d edge_edge_distance_hessian(
//     const glm::tvec3<Scalar>& ea0,
//     const glm::tvec3<Scalar>& ea1,
//     const glm::tvec3<Scalar>& eb0,
//     const glm::tvec3<Scalar>& eb1,
//     DistanceType dtype = DistanceType::AUTO);

} // namespace ipc
