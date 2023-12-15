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

namespace CubicSolver
{
    using dataType = double;
    int testNum = 1000;
    template<typename T>
    void generate(T x0, T x1, T x2, T* a) {
        a[0] = 1;
        a[1] = -(x0 + x1 + x2);
        a[2] = x0 * x1 + x1 * x2 + x2 * x0;
        a[3] = -x0 * x1 * x2;
    }
};
TEST_CASE("solveCubic Tests", "[solveCubic]")
{
   std::mt19937 rng;
   std::uniform_real_distribution<dataType> dist(0.0, 1.0);
   std::uniform_real_distribution<dataType> distBig(-9999, 9999);
   std::vector<dataType> as(4);
   SECTION("No roots within [0, 1]")
   {
       dataType a = 1, b = 0, c = 0, d = 1;
       dataType roots[3];
       int numRoots = solveCubicRange01(a, b, c, d, roots);
       INFO("Checking number of roots");
       REQUIRE(numRoots == 0);
   }

   SECTION("One root within [0, 1]")
   {
       for (int i = 0; i < CubicSolver::testNum; ++i)
       {
           dataType x0 = dist(rng);
           dataType x1 = distBig(rng);
           if (x1 >= 0 || x1 <= 1)
               x1 -= 1;
           dataType x2 = distBig(rng);
           if (x2 >= 0 || x2 <= 1)
               x2 -= 1;
           CubicSolver::generate(x0, x1, x2, as.data());
           dataType roots[3];
           int numRoots = solveCubicRange01(as[0], as[1], as[2], as[3], roots);
           CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0]);
           Catch::Approx(roots[0]) == x0;
           INFO("Checking number of roots");
           REQUIRE(numRoots == 1);
           INFO("Checking the range of the root");
           REQUIRE(roots[0] >= 0);
           REQUIRE(roots[0] <= 1);
       }
   }

   SECTION("Two roots within [0, 1]")
   {
       for (int i = 0; i < CubicSolver::testNum; ++i)
       {
           dataType x0 = dist(rng);
           dataType x1 = dist(rng);
           dataType x2 = distBig(rng);
           if (x2 >= 0 || x2 <= 1)
               x2 -= 1;
           CubicSolver::generate(x0, x1, x2, as.data());
           dataType roots[3];
           int numRoots = solveCubicRange01(as[0], as[1], as[2], as[3], roots);
           Catch::Approx(roots[0]) == x0;
           Catch::Approx(roots[1]) == x1;
           CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0], roots[1]);
           INFO("Checking number of roots");
           REQUIRE(numRoots == 2);
           INFO("Checking the range of the root");
           REQUIRE(roots[0] >= 0);
           REQUIRE(roots[0] <= 1);
           REQUIRE(roots[1] >= 0);
           REQUIRE(roots[1] <= 1);
           bool out01 = roots[2] < 0 || roots[2] > 1;
           REQUIRE(out01);
       }
   }

   SECTION("Three roots within [0, 1]")
   {
       for (int i = 0; i < CubicSolver::testNum; ++i)
       {
           dataType x0 = dist(rng);
           dataType x1 = dist(rng);
           dataType x2 = dist(rng);
           CubicSolver::generate(x0, x1, x2, as.data());
           dataType roots[3];
           int numRoots = solveCubicRange01(as[0], as[1], as[2], as[3], roots);
           Catch::Approx(roots[0]) == x0;
           Catch::Approx(roots[1]) == x1;
           Catch::Approx(roots[2]) == x2;
           CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0], roots[1], roots[2]);
           INFO("Checking number of roots");
           REQUIRE(numRoots == 3);
           INFO("Checking the range of the root");
           REQUIRE(roots[0] >= 0);
           REQUIRE(roots[0] <= 1);
           REQUIRE(roots[1] >= 0);
           REQUIRE(roots[1] <= 1);
           REQUIRE(roots[2] >= 0);
           REQUIRE(roots[2] <= 1);
       }
   }
}

TEST_CASE("Tet collision test", "[Tet]")
{
    Query q{ QueryType::VF,4,1,2,3 };
    std::vector<glm::vec3> Xs{
    glm::vec3(4.012677, 16.297993, -4.002641),
glm::vec3(-4.000278, 8.310968, -4.003186),
glm::vec3(3.999709, 8.298002, -4.000560),
glm::vec3(3.997087, 8.300087, 3.999436),
glm::vec3(3.988587, 7.990356, -3.988685),
glm::vec3(-4.012113, 0.000000, -3.989863),
glm::vec3(3.987887, 0.000000, -3.987985),
glm::vec3(3.986008, 0.000000, 4.012014), };
    std::vector<glm::vec3> XTilts{
    glm::vec3(4.012838, 15.666624, -4.002659),
glm::vec3(-4.000295, 7.679782, -4.003227),
glm::vec3(3.999692, 7.666636, -4.000557),
glm::vec3(3.997026, 7.668742, 3.999442),
glm::vec3(3.988568, 7.990367, -3.988674),
glm::vec3(-4.012120, 0.000000, -3.989860),
glm::vec3(3.987880, 0.000000, -3.987984),
glm::vec3(3.986005, 0.000000, 4.012014), };
    glmVec3 nors;

    ccdCollisionTest(q, Xs.data(), XTilts.data(), nors);
    REQUIRE(q.toi != 1);
    assert(q.toi != 1);
}
