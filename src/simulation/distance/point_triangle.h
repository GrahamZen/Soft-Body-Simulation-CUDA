#pragma once

#include <distance/distance_type.h>
#include <matrix.h>
#include <cuda_runtime.h>

namespace ipc {

/// @brief Compute the distance between a points and a triangle.
/// @note The distance is actually squared distance.
/// @param p The point.
/// @param t0 The first vertex of the triangle.
/// @param t1 The second vertex of the triangle.
/// @param t2 The third vertex of the triangle.
/// @param dtype The point-triangle distance type to compute.
/// @return The distance between the point and triangle.
template<typename Scalar>
__host__ __device__ Scalar point_triangle_distance(
    const glm::tvec3<Scalar>& p,
    const glm::tvec3<Scalar>& t0,
    const glm::tvec3<Scalar>& t1,
    const glm::tvec3<Scalar>& t2,
    DistanceType dtype);

/// @brief Compute the gradient of the distance between a points and a triangle.
/// @note The distance is actually squared distance.
/// @param p The point.
/// @param t0 The first vertex of the triangle.
/// @param t1 The second vertex of the triangle.
/// @param t2 The third vertex of the triangle.
/// @param dtype The point-triangle distance type to compute.
/// @return The gradient of the distance wrt p, t0, t1, and t2.
 template<typename Scalar>
 __host__ __device__ Vector12<Scalar> point_triangle_distance_gradient(
     const glm::tvec3<Scalar>& p,
     const glm::tvec3<Scalar>& t0,
     const glm::tvec3<Scalar>& t1,
     const glm::tvec3<Scalar>& t2,
     DistanceType dtype = DistanceType::AUTO);

 /// @brief Compute the hessian of the distance between a points and a triangle.
 /// @note The distance is actually squared distance.
 /// @param p The point.
 /// @param t0 The first vertex of the triangle.
 /// @param t1 The second vertex of the triangle.
 /// @param t2 The third vertex of the triangle.
 /// @param dtype The point-triangle distance type to compute.
 /// @return The hessian of the distance wrt p, t0, t1, and t2.
 template<typename Scalar>
 __host__ __device__ Matrix12<Scalar> point_triangle_distance_hessian(
     const glm::tvec3<Scalar>& p,
     const glm::tvec3<Scalar>& t0,
     const glm::tvec3<Scalar>& t1,
     const glm::tvec3<Scalar>& t2,
     DistanceType dtype = DistanceType::AUTO);

} // namespace ipc
