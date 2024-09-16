#pragma once

#include <matrix.h>
#include<glm/glm.hpp>

namespace ipc {

    /// @brief Compute the distance between a point and line in 2D or 3D.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param e0 The first vertex of the edge defining the line.
    /// @param e1 The second vertex of the edge defining the line.
    /// @return The distance between the point and line.
    template<typename Scalar>
    __host__ __device__ Scalar point_line_distance(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& e0,
        const glm::tvec3<Scalar>& e1);

    /// @brief Compute the gradient of the distance between a point and line.
    /// @note The distance is actually squared distance.
    /// @param p The point.
    /// @param e0 The first vertex of the edge defining the line.
    /// @param e1 The second vertex of the edge defining the line.
    /// @return The gradient of the distance wrt p, e0, and e1.
    template<typename Scalar>
    __host__ __device__ Vector<Scalar, 9> point_line_distance_gradient(
        const glm::tvec3<Scalar>& p,
        const glm::tvec3<Scalar>& e0,
        const glm::tvec3<Scalar>& e1);

    // /// @brief Compute the hessian of the distance between a point and line.
    // /// @note The distance is actually squared distance.
    // /// @param p The point.
    // /// @param e0 The first vertex of the edge defining the line.
    // /// @param e1 The second vertex of the edge defining the line.
    // /// @return The hessian of the distance wrt p, e0, and e1.
    // template<typename Scalar>
    // MatrixMax9d point_line_distance_hessian(
    //     const glm::tvec3<Scalar>& p,
    //     const glm::tvec3<Scalar>& e0,
    //     const glm::tvec3<Scalar>& e1);

    // Symbolically generated derivatives;
    namespace autogen {
        template<typename Scalar>
        void point_line_distance_gradient_2D(
            Scalar v01,
            Scalar v02,
            Scalar v11,
            Scalar v12,
            Scalar v21,
            Scalar v22,
            Scalar g[6]);

        template<typename Scalar>
        void point_line_distance_gradient_3D(
            Scalar v01,
            Scalar v02,
            Scalar v03,
            Scalar v11,
            Scalar v12,
            Scalar v13,
            Scalar v21,
            Scalar v22,
            Scalar v23,
            Scalar g[9]);

        template<typename Scalar>
        void point_line_distance_hessian_2D(
            Scalar v01,
            Scalar v02,
            Scalar v11,
            Scalar v12,
            Scalar v21,
            Scalar v22,
            Scalar H[36]);

        template<typename Scalar>
        void point_line_distance_hessian_3D(
            Scalar v01,
            Scalar v02,
            Scalar v03,
            Scalar v11,
            Scalar v12,
            Scalar v13,
            Scalar v21,
            Scalar v22,
            Scalar v23,
            Scalar H[81]);
    } // namespace autogen
} // namespace ipc
