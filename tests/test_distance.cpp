#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/cg.h>
#include <distance/distance_type.h>
#include <matrix.h>
#include <finitediff.hpp>

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar barrierSquareFunc(Scalar d_sqr, Scalar dhat, Scalar kappa) {
    Scalar s = d_sqr / (dhat * dhat);
    return 0.5 * dhat * kappa * 0.125 * (s - 1) * log(s);
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar barrierSquareFuncDerivate(Scalar d_sqr, Scalar dhat, Scalar kappa) {
    Scalar dhat_sqr = dhat * dhat;
    Scalar s = d_sqr / dhat_sqr;
    return 0.5 * dhat * (kappa / 8 * (log(s) / dhat_sqr + (s - 1) / d_sqr));
}

template <typename Scalar>
__forceinline__ __host__ __device__ Matrix12<Scalar> barrierSquareFuncHess(Scalar d_sqr, Scalar dhat, Scalar kappa, const Vector12<Scalar>& d_sqr_grad, const Matrix12<Scalar>& d_sqr_hess) {
    Scalar dhat_sqr = dhat * dhat;
    Scalar s = d_sqr / dhat_sqr;
    return 0.5 * dhat * (kappa / (8 * d_sqr * d_sqr * dhat_sqr) * (d_sqr + dhat_sqr) * Matrix12<Scalar>(d_sqr_grad, d_sqr_grad)
        + (kappa / 8 * (log(s) / dhat_sqr + (s - 1) / d_sqr)) * d_sqr_hess);
}

TEST_CASE("barrier", "[BARRIER][.][SKIP]") {
    std::vector<Query> queries = {
        Query{QueryType::VF,DistanceType::P_T,0.00068555,4,1,2,3,1,glm::dvec3(0.000000, 0.000000, 0.000000)}
    };
    std::vector<glm::dvec3> points = {
        glm::dvec3(1.469664, 15.174376, -0.742432),
        glm::dvec3(-1.466494, 5.704086, -6.191541),
        glm::dvec3(5.464619, 9.165228, -4.196486),
        glm::dvec3(5.460415, 5.176405, 2.738156),
        glm::dvec3(4.001574, 8.045403, -4.001087),
        glm::dvec3(-4.000312, 0.048792, -4.000341),
        glm::dvec3(3.999855, 0.047073, -3.999369),
        glm::dvec3(3.998883, 0.048791, 4.000798)
    };
    for (size_t i = 0; i < queries.size(); i++)
    {
        auto& q = queries[i];
        glm::dvec3 x0 = points[q.v0], x1 = points[q.v1], x2 = points[q.v2], x3 = points[q.v3];
        Vector12<double> grad;
        Matrix12<double> hess;
        if (q.type == QueryType::VF) {
            if (q.dType == DistanceType::AUTO)
                q.dType = point_triangle_distance_type(x0, x1, x2, x3);
            q.d = ipc::point_triangle_distance(x0, x1, x2, x3, q.dType);
            grad = ipc::point_triangle_distance_gradient<double>(x0, x1, x2, x3, q.dType);
            hess = ipc::point_triangle_distance_hessian<double>(x0, x1, x2, x3, q.dType);

            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> x;
            x << x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z, x3.x, x3.y, x3.z;
            // *******************************************************************************
            // Compare the gradient with finite differences
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> eig_grad;
            eig_grad << grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9], grad[10], grad[11];

            Eigen::VectorXd fgrad(1);
            fd::finite_gradient(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                double d = ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
                return d;
                }
            , fgrad);
            CHECK(fd::compare_gradient(eig_grad, fgrad));
            // *******************************************************************************
            // Compare the hessian with finite differences
            Eigen::MatrixXd eig_hess(12, 12);
            eig_hess = Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>>(hess.data());
            Eigen::MatrixXd fhess(12, 12);
            fd::finite_hessian(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                return (double)ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
                }
            , fhess);
            CHECK(fd::compare_hessian(eig_hess, fhess));
        }
        else if (q.type == QueryType::EE) {
            if (q.dType == DistanceType::AUTO)
                q.dType = edge_edge_distance_type(x0, x1, x2, x3);
            q.d = ipc::edge_edge_distance(x0, x1, x2, x3, q.dType);
            grad = ipc::edge_edge_distance_gradient<double>(x0, x1, x2, x3, q.dType);
            hess = ipc::edge_edge_distance_hessian<double>(x0, x1, x2, x3, q.dType);
            // *******************************************************************************
            // Compare the gradient with finite differences
            Eigen::VectorXd fgrad(1);
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> x;
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> eig_grad;
            eig_grad << grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9], grad[10], grad[11];
            x << x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z, x3.x, x3.y, x3.z;

            // Compute the gradient using finite differences
            fd::finite_gradient(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                return (double)ipc::edge_edge_distance(_x0, _x1, _x2, _x3, q.dType);
                }
            , fgrad);
            CHECK(fd::compare_gradient(eig_grad, fgrad));
            // *******************************************************************************
            // Compare the hessian with finite differences
            Eigen::MatrixXd eig_hess(12, 12);
            eig_hess = Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>>(hess.data());
            Eigen::MatrixXd fhess(12, 12);
            fd::finite_hessian(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                return (double)ipc::edge_edge_distance(_x0, _x1, _x2, _x3, q.dType);
                }
            , fhess);
            CHECK(fd::compare_hessian(eig_hess, fhess));
        }
        printVector(grad, "grad");
        printMatrix(hess, "hess");
    }
}

TEST_CASE("distance", "[DISTANCE]") {
    double dhat = 0.05;
    double kappa = 100;
    std::vector<Query> queries = {
        Query{QueryType::VF,DistanceType::P_T,0.00068555,4,1,2,3,1,glm::dvec3(0.000000, 0.000000, 0.000000)}
    };
    std::vector<glm::dvec3> points = {
        glm::dvec3(1.469664, 15.174376, -0.742432),
        glm::dvec3(-1.466494, 5.704086, -6.191541),
        glm::dvec3(5.464619, 9.165228, -4.196486),
        glm::dvec3(5.460415, 5.176405, 2.738156),
        glm::dvec3(4.001574, 8.045403, -4.001087),
        glm::dvec3(-4.000312, 0.048792, -4.000341),
        glm::dvec3(3.999855, 0.047073, -3.999369),
        glm::dvec3(3.998883, 0.048791, 4.000798)
    };
    for (size_t i = 0; i < queries.size(); i++)
    {
        auto& q = queries[i];
        glm::dvec3 x0 = points[q.v0], x1 = points[q.v1], x2 = points[q.v2], x3 = points[q.v3];
        Vector12<double> grad;
        Matrix12<double> hess;
        if (q.type == QueryType::VF) {
            if (q.dType == DistanceType::AUTO)
                q.dType = point_triangle_distance_type(x0, x1, x2, x3);
            q.d = ipc::point_triangle_distance(x0, x1, x2, x3, q.dType);
            Vector12<double>local_grad = ipc::point_triangle_distance_gradient<double>(x0, x1, x2, x3, q.dType);
            double der = barrierSquareFuncDerivate((double)q.d, dhat, kappa);
            grad = der * local_grad;
            hess = ipc::point_triangle_distance_hessian<double>(x0, x1, x2, x3, q.dType);
            hess = barrierSquareFuncHess(double(q.d), dhat, kappa, local_grad, hess);
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> x;
            x << x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z, x3.x, x3.y, x3.z;
            // *******************************************************************************
            // Compare the gradient with finite differences
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> eig_grad;
            eig_grad = Eigen::Map<Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1>>(grad.data());

            Eigen::VectorXd fgrad(1);
            fd::finite_gradient(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                double d = ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
                return barrierSquareFunc(d, dhat, kappa);
                }
            , fgrad);

            double bgrad1 = barrierSquareFuncDerivate((double)q.d, dhat, kappa);
            CHECK(fd::compare_gradient(eig_grad, fgrad));
            // *******************************************************************************
            // Compare the hessian with finite differences
            Eigen::MatrixXd eig_hess(12, 12);
            eig_hess = Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>>(hess.data());
            Eigen::MatrixXd fhess(12, 12);
            fd::finite_hessian(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                double d = ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
                return barrierSquareFunc(d, dhat, kappa);
                }
            , fhess);
            CHECK(fd::compare_hessian(eig_hess, fhess));
        }
        else if (q.type == QueryType::EE) {
            if (q.dType == DistanceType::AUTO)
                q.dType = edge_edge_distance_type(x0, x1, x2, x3);
            q.d = ipc::edge_edge_distance(x0, x1, x2, x3, q.dType);
            grad = ipc::edge_edge_distance_gradient<double>(x0, x1, x2, x3, q.dType);
            hess = ipc::edge_edge_distance_hessian<double>(x0, x1, x2, x3, q.dType);
            // *******************************************************************************
            // Compare the gradient with finite differences
            Eigen::VectorXd fgrad(1);
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> x;
            Eigen::Matrix<double, 12, 1, Eigen::ColMajor, 12, 1> eig_grad;
            eig_grad << grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9], grad[10], grad[11];
            x << x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z, x3.x, x3.y, x3.z;

            // Compute the gradient using finite differences
            fd::finite_gradient(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                double d = ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
                return barrierSquareFunc(d, dhat, kappa);
                }
            , fgrad);
            CHECK(fd::compare_gradient(eig_grad, fgrad));
            // *******************************************************************************
            // Compare the hessian with finite differences
            Eigen::MatrixXd eig_hess(12, 12);
            eig_hess = Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>>(hess.data());
            Eigen::MatrixXd fhess(12, 12);
            fd::finite_hessian(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                double d = ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
                return barrierSquareFunc(d, dhat, kappa);
                }
            , fhess);
            CHECK(fd::compare_hessian(eig_hess, fhess));
        }
        printVector(grad, "grad");
        printMatrix(hess, "hess");
    }
}