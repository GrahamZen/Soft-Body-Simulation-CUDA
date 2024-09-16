#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <linear/cg.h>
#include <distance/distance_type.h>
#include <matrix.h>
#include <finitediff.hpp>


TEST_CASE("distance", "[DISTANCE]") {
    std::vector<Query> queries = {
        Query{QueryType::EE,DistanceType::AUTO,0,0,3,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,4,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,3,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,4,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,0,3},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,0,4},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,1,3},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,1,4},
        Query{QueryType::VF,DistanceType::AUTO,0,2,1,3,5},
        Query{QueryType::VF,DistanceType::AUTO,0,2,1,4,7},
        Query{QueryType::VF,DistanceType::AUTO,0,6,0,3,5},
        Query{QueryType::VF,DistanceType::AUTO,0,6,0,4,7},
        Query{QueryType::VF,DistanceType::AUTO,0,3,2,4,6},
        Query{QueryType::VF,DistanceType::AUTO,0,4,2,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,3,2,4},
        Query{QueryType::EE,DistanceType::AUTO,0,1,4,2,3},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,3,5},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,4,7},
        Query{QueryType::EE,DistanceType::AUTO,0,0,3,4,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,4,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,2,3,1,4},
        Query{QueryType::EE,DistanceType::AUTO,0,2,4,1,3},
        Query{QueryType::EE,DistanceType::AUTO,0,3,5,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,4,7,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,3,6,0,4},
        Query{QueryType::EE,DistanceType::AUTO,0,4,6,0,3},
        Query{QueryType::VF,DistanceType::AUTO,0,1,2,3,6},
        Query{QueryType::VF,DistanceType::AUTO,0,1,2,4,6},
        Query{QueryType::VF,DistanceType::AUTO,0,2,0,3,5},
        Query{QueryType::VF,DistanceType::AUTO,0,2,0,4,7},
        Query{QueryType::VF,DistanceType::AUTO,0,6,1,3,5},
        Query{QueryType::VF,DistanceType::AUTO,0,6,1,4,7},
        Query{QueryType::VF,DistanceType::AUTO,0,0,2,3,6},
        Query{QueryType::VF,DistanceType::AUTO,0,0,2,4,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,5,2,4},
        Query{QueryType::EE,DistanceType::AUTO,0,1,5,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,7,2,3},
        Query{QueryType::EE,DistanceType::AUTO,0,1,7,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,2,3,1,7},
        Query{QueryType::EE,DistanceType::AUTO,0,2,3,4,7},
        Query{QueryType::EE,DistanceType::AUTO,0,2,4,1,5},
        Query{QueryType::EE,DistanceType::AUTO,0,2,4,3,5},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,1,5},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,1,7},
        Query{QueryType::EE,DistanceType::AUTO,0,3,5,2,4},
        Query{QueryType::EE,DistanceType::AUTO,0,4,7,2,3},
        Query{QueryType::EE,DistanceType::AUTO,0,0,5,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,5,4,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,7,2,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,7,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,2,3,0,4},
        Query{QueryType::EE,DistanceType::AUTO,0,2,4,0,3},
        Query{QueryType::EE,DistanceType::AUTO,0,3,5,4,6},
        Query{QueryType::EE,DistanceType::AUTO,0,4,7,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,3,2,4},
        Query{QueryType::EE,DistanceType::AUTO,0,0,4,2,3},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,0,5},
        Query{QueryType::EE,DistanceType::AUTO,0,2,6,0,7},
        Query{QueryType::EE,DistanceType::AUTO,0,3,6,0,7},
        Query{QueryType::EE,DistanceType::AUTO,0,3,6,4,7},
        Query{QueryType::EE,DistanceType::AUTO,0,4,6,0,5},
        Query{QueryType::EE,DistanceType::AUTO,0,4,6,3,5},
        Query{QueryType::EE,DistanceType::AUTO,0,1,3,4,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,4,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,3,6,1,4},
        Query{QueryType::EE,DistanceType::AUTO,0,4,6,1,3},
        Query{QueryType::VF,DistanceType::AUTO,0,3,0,4,7},
        Query{QueryType::VF,DistanceType::AUTO,0,3,1,4,7},
        Query{QueryType::VF,DistanceType::AUTO,0,4,0,3,5},
        Query{QueryType::VF,DistanceType::AUTO,0,4,1,3,5},
        Query{QueryType::VF,DistanceType::AUTO,0,5,2,4,6},
        Query{QueryType::VF,DistanceType::AUTO,0,7,2,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,0,5,2,4},
        Query{QueryType::EE,DistanceType::AUTO,0,0,7,2,3},
        Query{QueryType::EE,DistanceType::AUTO,0,1,5,4,6},
        Query{QueryType::EE,DistanceType::AUTO,0,1,7,3,6},
        Query{QueryType::EE,DistanceType::AUTO,0,2,3,0,7},
        Query{QueryType::EE,DistanceType::AUTO,0,2,4,0,5},
        Query{QueryType::EE,DistanceType::AUTO,0,3,6,1,7},
        Query{QueryType::EE,DistanceType::AUTO,0,4,6,1,5},
    };
    std::vector<glm::dvec3> points = {
        glm::dvec3(-1.000000, 51.000000, 1.000000),
        glm::dvec3(-1.000000, 49.000000, -1.000000),
        glm::dvec3(1.000000, 49.000000, -1.000000),
        glm::dvec3(1.000000, 51.000000, -1.000000),
        glm::dvec3(1.000000, 49.000000, 1.000000),
        glm::dvec3(-1.000000, 51.000000, -1.000000),
        glm::dvec3(1.000000, 51.000000, 1.000000),
        glm::dvec3(-1.000000, 49.000000, 1.000000)
    };
    for (size_t i = 0; i < queries.size(); i++)
    {
        auto& q = queries[i];
        glm::dvec3 x0 = points[q.v0], x1 = points[q.v1], x2 = points[q.v2], x3 = points[q.v3];
        Vector12<double> grad;
        Matrix12<double> hess;
        if (q.type == QueryType::VF) {
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
                return (double)ipc::point_triangle_distance(_x0, _x1, _x2, _x3, q.dType);
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
