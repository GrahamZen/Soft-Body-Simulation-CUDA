#include <cuda_runtime.h>
#include <intersections.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/intersect.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
//
//namespace CubicSolver
//{
//    using dataType = double;
//    int testNum = 1000;
//    template<typename T>
//    void generate(T x0, T x1, T x2, T* a) {
//        a[0] = 1;
//        a[1] = -(x0 + x1 + x2);
//        a[2] = x0 * x1 + x1 * x2 + x2 * x0;
//        a[3] = -x0 * x1 * x2;
//    }
//};
//TEST_CASE("solveCubic Tests", "[solveCubic]")
//{
//   std::mt19937 rng;
//   std::uniform_real_distribution<dataType> dist(0.0, 1.0);
//   std::uniform_real_distribution<dataType> distBig(-9999, 9999);
//   std::vector<dataType> as(4);
//   SECTION("No roots within [0, 1]")
//   {
//       dataType a = 1, b = 0, c = 0, d = 1;
//       dataType roots[3];
//       int numRoots = solveCubicRange01(a, b, c, d, roots);
//       INFO("Checking number of roots");
//       REQUIRE(numRoots == 0);
//   }
//   SECTION("Three identical roots 1")
//   {
//       CubicSolver::generate<double>(2, 2, 2, as.data());
//       dataType roots[3];
//       int numRoots = solveCubic(as[0], as[1], as[2], as[3], roots);
//       INFO("Checking number of roots");
//       REQUIRE(numRoots == 3);
//   }
//
//   SECTION("One root within [0, 1]")
//   {
//       for (int i = 0; i < CubicSolver::testNum; ++i)
//       {
//           dataType x0 = dist(rng);
//           dataType x1 = distBig(rng);
//           if (x1 >= 0 || x1 <= 1)
//               x1 -= 1;
//           dataType x2 = distBig(rng);
//           if (x2 >= 0 || x2 <= 1)
//               x2 -= 1;
//           CubicSolver::generate(x0, x1, x2, as.data());
//           dataType roots[3];
//           int numRoots = solveCubicRange01(as[0], as[1], as[2], as[3], roots);
//           CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0]);
//           Catch::Approx(roots[0]) == x0;
//           INFO("Checking number of roots");
//           REQUIRE(numRoots == 1);
//           INFO("Checking the range of the root");
//           REQUIRE(roots[0] >= 0);
//           REQUIRE(roots[0] <= 1);
//       }
//   }
//
//   SECTION("Two roots within [0, 1]")
//   {
//       for (int i = 0; i < CubicSolver::testNum; ++i)
//       {
//           dataType x0 = dist(rng);
//           dataType x1 = dist(rng);
//           dataType x2 = distBig(rng);
//           if (x2 >= 0 || x2 <= 1)
//               x2 -= 1;
//           CubicSolver::generate(x0, x1, x2, as.data());
//           dataType roots[3];
//           int numRoots = solveCubicRange01(as[0], as[1], as[2], as[3], roots);
//           Catch::Approx(roots[0]) == x0;
//           Catch::Approx(roots[1]) == x1;
//           CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0], roots[1]);
//           INFO("Checking number of roots");
//           REQUIRE(numRoots == 2);
//           INFO("Checking the range of the root");
//           REQUIRE(roots[0] >= 0);
//           REQUIRE(roots[0] <= 1);
//           REQUIRE(roots[1] >= 0);
//           REQUIRE(roots[1] <= 1);
//           bool out01 = roots[2] < 0 || roots[2] > 1;
//           REQUIRE(out01);
//       }
//   }
//
//   SECTION("Three roots within [0, 1]")
//   {
//       for (int i = 0; i < CubicSolver::testNum; ++i)
//       {
//           dataType x0 = dist(rng);
//           dataType x1 = dist(rng);
//           dataType x2 = dist(rng);
//           CubicSolver::generate(x0, x1, x2, as.data());
//           dataType roots[3];
//           int numRoots = solveCubicRange01(as[0], as[1], as[2], as[3], roots);
//           Catch::Approx(roots[0]) == x0;
//           Catch::Approx(roots[1]) == x1;
//           Catch::Approx(roots[2]) == x2;
//           CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0], roots[1], roots[2]);
//           INFO("Checking number of roots");
//           REQUIRE(numRoots == 3);
//           INFO("Checking the range of the root");
//           REQUIRE(roots[0] >= 0);
//           REQUIRE(roots[0] <= 1);
//           REQUIRE(roots[1] >= 0);
//           REQUIRE(roots[1] <= 1);
//           REQUIRE(roots[2] >= 0);
//           REQUIRE(roots[2] <= 1);
//       }
//   }
//}

TEST_CASE("Tet collision test", "[Tet]")
{
    std::vector<Query> queries = {
Query{QueryType::VF,DistanceType::P_T0,0,4,5,9,1,0,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::VF,DistanceType::P_T0,1,4,5,9,1,0,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::VF,DistanceType::P_T0,2,4,5,9,1,0,Vec3d{0.000000, 0.000000, 0.000000}},
Query{QueryType::VF,DistanceType::EA1_EB0,3,4,7,10,0.994723,100,Vec3d{0.000010, 1.000000, 0.000010}},
Query{QueryType::VF,DistanceType::P_T0,4,0,1,3,1,0,Vec3d{0.000000, 0.000000, 0.000000}},
    };
    std::vector<glm::dvec3> Xs{
glm::dvec3(2.001631, 67.999075, 0.533407),
glm::dvec3(-6.001968, 61.071146, -3.459837),
glm::dvec3(1.998028, 61.070165, -3.465346),
glm::dvec3(2.002309, 57.071411, 3.463573),
glm::dvec3(-5.000000, 55.000000, 5.000000),
glm::dvec3(-5.000000, 45.000000, -5.000000),
glm::dvec3(5.000000, 45.000000, -5.000000),
glm::dvec3(5.000000, 55.000000, -5.000000),
glm::dvec3(4.999979, 44.999900, 4.999969),
glm::dvec3(-5.000017, 54.999918, -5.000016),
glm::dvec3(5.000011, 54.999895, 5.000018),
glm::dvec3(-5.000012, 44.999895, 4.999962),
    };
    std::vector<glm::dvec3> XTilts{
glm::dvec3(2.001704, 65.916575, 0.533288),
glm::dvec3(-6.002013, 58.988661, -3.459746),
glm::dvec3(1.997986, 58.987607, -3.465362),
glm::dvec3(2.002323, 54.988955, 3.463617),
glm::dvec3(-5.000000, 55.000000, 5.000000),
glm::dvec3(-5.000000, 45.000000, -5.000000),
glm::dvec3(5.000000, 45.000000, -5.000000),
glm::dvec3(5.000000, 55.000000, -5.000000),
glm::dvec3(4.999979, 44.999902, 4.999972),
glm::dvec3(-5.000016, 54.999917, -5.000015),
glm::dvec3(5.000011, 54.999896, 5.000015),
glm::dvec3(-5.000010, 44.999895, 4.999963),
    };
    glm::dvec3 nors;
    double toi = 1;
    for (size_t i = 0; i < queries.size(); i++)
    {
        auto& q = queries[i];
        q.toi = ccdCollisionTest(q, Xs.data(), XTilts.data(), nors);
        toi = std::min(toi, q.toi);
    }
    REQUIRE(toi != 1);
}
