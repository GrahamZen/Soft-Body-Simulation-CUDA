#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/cg.h>
#include <distance/distance_type.cuh>
#include <matrix.h>
#include <finitediff.hpp>

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar barrierSquareFunc(Scalar d_sqr, Scalar dhat, Scalar kappa) {
    Scalar s = d_sqr / (dhat * dhat);
    return 0.5 * dhat * kappa * 0.125 * (s - 1) * log(s);
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar barrierSquareFuncDerivative(Scalar d_sqr, Scalar dhat, Scalar kappa) {
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

double dhat = 0.05;
double kappa = 100;
std::vector<Query> queries = {
Query{QueryType::EE,DistanceType::EA_EB,1,2,4,7,0.00187279,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::EE,DistanceType::EA_EB,4,7,1,2,0.00187279,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::EE,DistanceType::EA_EB,1,2,4,5,0.0021027,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::EE,DistanceType::EA_EB,4,5,1,2,0.0021027,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::VF,DistanceType::P_E0,4,1,2,3,0.00265943,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::VF,DistanceType::P_E1,4,0,1,2,0.00265943,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::EE,DistanceType::EA_EB0,1,2,4,6,0.00265943,1,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::EE,DistanceType::EA0_EB,4,6,1,2,0.00265943,1,Vec3d{0.000000, 0.000000, 0.000000}},
};
std::vector<glm::dvec3> points = {
glm::dvec3(0.436417, 59.751547, 0.282352),
glm::dvec3(-1.886212, 50.571970, -5.875603),
glm::dvec3(4.767792, 54.469072, -3.694910),
glm::dvec3(5.295233, 49.857338, 2.850450),
glm::dvec3(4.000000, 54.000000, -4.000000),
glm::dvec3(-4.000000, 46.000000, -4.000000),
glm::dvec3(4.000000, 46.000000, -4.000000),
glm::dvec3(4.000000, 46.000000, 4.000000),
};

void test_distance(std::vector<Query>& queries, const std::vector<glm::dvec3>& points) {
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
    }
}

void test_barrier(std::vector<Query>& queries, const std::vector<glm::dvec3>& points) {
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
            grad = barrierSquareFuncDerivative((double)q.d, dhat, kappa) * local_grad;
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

            double bgrad1 = barrierSquareFuncDerivative((double)q.d, dhat, kappa);
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
            auto diff = (eig_hess - fhess).eval();
            CHECK(fd::compare_hessian(eig_hess, fhess));
        }
        else if (q.type == QueryType::EE) {
            if (q.dType == DistanceType::AUTO)
                q.dType = point_triangle_distance_type(x0, x1, x2, x3);
            q.d = ipc::edge_edge_distance(x0, x1, x2, x3, q.dType);
            Vector12<double>local_grad = ipc::edge_edge_distance_gradient<double>(x0, x1, x2, x3, q.dType);
            grad = barrierSquareFuncDerivative((double)q.d, dhat, kappa) * local_grad;
            hess = ipc::edge_edge_distance_hessian<double>(x0, x1, x2, x3, q.dType);
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
                double d = ipc::edge_edge_distance(_x0, _x1, _x2, _x3, q.dType);
                return barrierSquareFunc(d, dhat, kappa);
                }
            , fgrad);

            double bgrad1 = barrierSquareFuncDerivative((double)q.d, dhat, kappa);
            CHECK(fd::compare_gradient(eig_grad, fgrad));
            // *******************************************************************************
            // Compare the hessian with finite differences
            Eigen::MatrixXd eig_hess(12, 12);
            eig_hess = Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>>(hess.data());
            Eigen::MatrixXd fhess(12, 12);
            fd::finite_hessian(x, [=](const Eigen::VectorXd& _x) {
                glm::dvec3 _x0(_x[0], _x[1], _x[2]), _x1(_x[3], _x[4], _x[5]), _x2(_x[6], _x[7], _x[8]), _x3(_x[9], _x[10], _x[11]);
                double d = ipc::edge_edge_distance(_x0, _x1, _x2, _x3, q.dType);
                return barrierSquareFunc(d, dhat, kappa);
                }
            , fhess);
            auto diff = (eig_hess - fhess).eval();
            CHECK(fd::compare_hessian(eig_hess, fhess));
        }
    }
}

TEST_CASE("distance", "[DISTANCE]") {
    test_distance(queries, points);
}

TEST_CASE("barrier", "[BARRIER]") {
    test_barrier(queries, points);
}