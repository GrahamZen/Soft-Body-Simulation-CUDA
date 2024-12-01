#include "point_point.h"
#include <glm/gtx/norm.hpp> 
#include <cuda_runtime.h>

namespace ipc {

    template<typename Scalar>
    __host__ __device__ Scalar point_point_distance(
        const glm::tvec3<Scalar>& p0,
        const glm::tvec3<Scalar>& p1)
    {
        return glm::length2(p0 - p1);
    }

    template __host__ __device__ float point_point_distance<float>(
        const glm::tvec3<float>& p0,
        const glm::tvec3<float>& p1);

    template __host__ __device__ double point_point_distance<double>(
        const glm::tvec3<double>& p0,
        const glm::tvec3<double>& p1);

    template<typename Scalar>
    __host__ __device__ Vector<Scalar, 6> point_point_distance_gradient(
        const glm::tvec3<Scalar>& p0,
        const glm::tvec3<Scalar>& p1)
    {
        auto v = (Scalar)2.0 * (p0 - p1);
        Vector<Scalar, 6> grad;
        Vector<Scalar, 3> tmp = v;
        Vector<Scalar, 3> tmpM = -v;

        grad.head(3) = tmp;
        grad.tail(3) = tmpM;

        return grad;
    }

    template __host__ __device__ Vector<float, 6> point_point_distance_gradient<float>(
        const glm::tvec3<float>& p0,
        const glm::tvec3<float>& p1);

    template __host__ __device__ Vector<double, 6> point_point_distance_gradient<double>(
        const glm::tvec3<double>& p0,
        const glm::tvec3<double>& p1);

    template<typename Scalar>
    Matrix6<Scalar> __host__ __device__ point_point_distance_hessian(
        const glm::tvec3<Scalar>& p0,
        const glm::tvec3<Scalar>& p1)
    {
        Matrix6<Scalar> hess;
        hess[0][0] = hess[1][1] = hess[2][2] = hess[3][3] = hess[4][4] = hess[5][5] = 2.0;
        for (int i = 0; i < 3; i++) {
            hess[i][i + 3] = hess[i + 3][i] = -2.0;
        }
        return hess;
    }

    template Matrix6<float> __host__ __device__ point_point_distance_hessian<float>(
        const glm::tvec3<float>& p0,
        const glm::tvec3<float>& p1);

    template Matrix6<double> __host__ __device__ point_point_distance_hessian<double>(
        const glm::tvec3<double>& p0,
        const glm::tvec3<double>& p1);

} // namespace ipc
