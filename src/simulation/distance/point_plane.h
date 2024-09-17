#pragma once

#include <matrix.h>
#include<glm/glm.hpp>

namespace ipc {

    /// @brief Compute the distance between a point and a plane.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param origin The origin of the plane.
    /// @param normal The normal of the plane.
    /// @return The distance between the point and plane.
    template<typename Scalar>
    __host__ __device__ Scalar point_plane_distance(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& origin,
        const glm::tvec3<Scalar>& normal);

    /// @brief Compute the distance between a point and a plane.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param t0 The first vertex of the triangle.
    /// @param t1 The second vertex of the triangle.
    /// @param t2 The third vertex of the triangle.
    /// @return The distance between the point and plane.
    template<typename Scalar>
    __host__ __device__ Scalar point_plane_distance(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& t0,
        const glm::tvec3<Scalar>& t1,
        const glm::tvec3<Scalar>& t2);

    /// @brief Compute the gradient of the distance between a point and a plane.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param origin The origin of the plane.
    /// @param normal The normal of the plane.
    /// @return The gradient of the distance wrt p.
    template<typename Scalar>
    __host__ __device__ Vector<Scalar, 3> point_plane_distance_gradient(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& origin,
        const glm::tvec3<Scalar>& normal);

    /// @brief Compute the gradient of the distance between a point and a plane.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param t0 The first vertex of the triangle.
    /// @param t1 The second vertex of the triangle.
    /// @param t2 The third vertex of the triangle.
    /// @return The gradient of the distance wrt p, t0, t1, and t2.
    template<typename Scalar>
    __host__ __device__ Vector12<Scalar> point_plane_distance_gradient(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& t0,
        const glm::tvec3<Scalar>& t1,
        const glm::tvec3<Scalar>& t2);

    /// @brief Compute the hessian of the distance between a point and a plane.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param origin The origin of the plane.
    /// @param normal The normal of the plane.
    /// @return The hessian of the distance wrt p.
    template<typename Scalar>
    __host__ __device__ Matrix3<Scalar> point_plane_distance_hessian(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& origin,
        const glm::tvec3<Scalar>& normal);

    /// @brief Compute the hessian of the distance between a point and a plane.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param t0 The first vertex of the triangle.
    /// @param t1 The second vertex of the triangle.
    /// @param t2 The third vertex of the triangle.
    /// @return The hessian of the distance wrt p, t0, t1, and t2.
    template<typename Scalar>
    __host__ __device__ Matrix12<Scalar> point_plane_distance_hessian(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& t0,
        const glm::tvec3<Scalar>& t1,
        const glm::tvec3<Scalar>& t2);

    // Symbolically generated derivatives;
    namespace autogen {
        template<typename Scalar>
        __host__ __device__  void point_plane_distance_gradient(
            Scalar v01,
            Scalar v02,
            Scalar v03,
            Scalar v11,
            Scalar v12,
            Scalar v13,
            Scalar v21,
            Scalar v22,
            Scalar v23,
            Scalar v31,
            Scalar v32,
            Scalar v33,
            Scalar g[12]);

        template<typename Scalar>
        __host__ __device__ void point_plane_distance_hessian(
            Scalar v01,
            Scalar v02,
            Scalar v03,
            Scalar v11,
            Scalar v12,
            Scalar v13,
            Scalar v21,
            Scalar v22,
            Scalar v23,
            Scalar v31,
            Scalar v32,
            Scalar v33,
            Scalar H[144]);
    } // namespace autogen

} // namespace ipc
