#pragma once

#include <matrix.h>
#include<glm/glm.hpp>

namespace ipc {

    /// @brief Compute the distance between a two infinite lines in 3D.
    /// @note The distance is actually squared distance.
    /// @warning If the lines are parallel this function returns a distance of zero.
    /// @param ea0 The first vertex of the edge defining the first line.
    /// @param ea1 The second vertex of the edge defining the first line.
    /// @param ea0 The first vertex of the edge defining the second line.
    /// @param ea1 The second vertex of the edge defining the second line.
    /// @return The distance between the two lines.
    template<typename Scalar>
    __host__ __device__ Scalar line_line_distance(
        const glm::tvec3<Scalar>& ea0,
        const glm::tvec3<Scalar>& ea1,
        const glm::tvec3<Scalar>& eb0,
        const glm::tvec3<Scalar>& eb1);

    /// @brief Compute the gradient of the distance between a two lines in 3D.
    /// @note The distance is actually squared distance.
    /// @warning If the lines are parallel this function returns a distance of zero.
    /// @param ea0 The first vertex of the edge defining the first line.
    /// @param ea1 The second vertex of the edge defining the first line.
    /// @param ea0 The first vertex of the edge defining the second line.
    /// @param ea1 The second vertex of the edge defining the second line.
    /// @return The gradient of the distance wrt ea0, ea1, eb0, and eb1.
    template<typename Scalar>
    __host__ __device__ Vector12<Scalar> line_line_distance_gradient(
        const glm::tvec3<Scalar>& ea0,
        const glm::tvec3<Scalar>& ea1,
        const glm::tvec3<Scalar>& eb0,
        const glm::tvec3<Scalar>& eb1);

    /// @brief Compute the hessian of the distance between a two lines in 3D.
    /// @note The distance is actually squared distance.
    /// @warning If the lines are parallel this function returns a distance of zero.
    /// @param ea0 The first vertex of the edge defining the first line.
    /// @param ea1 The second vertex of the edge defining the first line.
    /// @param ea0 The first vertex of the edge defining the second line.
    /// @param ea1 The second vertex of the edge defining the second line.
    /// @return The hessian of the distance wrt ea0, ea1, eb0, and eb1.
    template<typename Scalar>
    Matrix12<Scalar> line_line_distance_hessian(
        const glm::tvec3<Scalar>& ea0,
        const glm::tvec3<Scalar>& ea1,
        const glm::tvec3<Scalar>& eb0,
        const glm::tvec3<Scalar>& eb1);

    // Symbolically generated derivatives;
    namespace autogen {
        template<typename Scalar>
        __host__ __device__ void line_line_distance_gradient(
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
        void line_line_distance_hessian(
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
