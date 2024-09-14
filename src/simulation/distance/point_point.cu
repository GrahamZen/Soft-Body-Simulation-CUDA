#include "point_point.h"
#include <glm/gtx/norm.hpp> 
#include <cuda_runtime.h>

namespace ipc {

    template<typename Scalar>
    __device__ Scalar point_point_distance(
        const glm::tvec3<Scalar>& p0,
        const glm::tvec3<Scalar>& p1)
    {
        return glm::length2(p0 - p1);
    }

    template __device__ float point_point_distance<float>(
        const glm::tvec3<float>& p0,
        const glm::tvec3<float>& p1);

    template __device__ double point_point_distance<double>(
        const glm::tvec3<double>& p0,
        const glm::tvec3<double>& p1);

    // template<typename Scalar>
    // VectorMax6d point_point_distance_gradient(
    //     const glm::tvec3<Scalar>& p0,
    //     const glm::tvec3<Scalar>& p1)
    // {
    //     int dim = p0.size();
    //     assert(p1.size() == dim);

    //     VectorMax6d grad(2 * dim);

    //     grad.head(dim) = 2.0 * (p0 - p1);
    //     grad.tail(dim) = -grad.head(dim);

    //     return grad;
    // }

    // template<typename Scalar>
    // MatrixMax6d point_point_distance_hessian(
    //     const glm::tvec3<Scalar>& p0,
    //     const glm::tvec3<Scalar>& p1)
    // {
    //     int dim = p0.size();
    //     assert(p1.size() == dim);

    //     MatrixMax6d hess(2 * dim, 2 * dim);

    //     hess.setZero();
    //     hess.diagonal().setConstant(2.0);
    //     for (int i = 0; i < dim; i++) {
    //         hess(i, i + dim) = hess(i + dim, i) = -2;
    //     }

    //     return hess;
    // }

} // namespace ipc
