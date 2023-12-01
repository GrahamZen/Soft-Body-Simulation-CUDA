#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>
#include <intersections.h>
#include <iostream>

namespace CubicSolver {
    using dataType = double;
    int testNum = 1000;
};

//TEST_CASE("solveCubic Tests", "[solveCubic]") {
//    std::mt19937 rng;
//    std::uniform_real_distribution<CubicSolver::dataType> dist(0.0, 1.0);
//    std::uniform_real_distribution<CubicSolver::dataType> distBig(-9999, 9999);
//    std::vector<CubicSolver::dataType>as(4);
//    SECTION("No roots within [0, 1]") {
//        CubicSolver::dataType a = 1, b = 0, c = 0, d = 1;
//        CubicSolver::dataType roots[3];
//        int numRoots = CubicSolver::solveCubicRange01(a, b, c, d, roots);
//        INFO("Checking number of roots");
//        REQUIRE(numRoots == 0);
//    }
//
//    SECTION("One root within [0, 1]") {
//        for (int i = 0; i < CubicSolver::testNum; ++i) {
//            CubicSolver::dataType x0 = dist(rng);
//            CubicSolver::dataType x1 = distBig(rng);
//            if (x1 >= 0 || x1 <= 1)x1 -= 1;
//            CubicSolver::dataType x2 = distBig(rng);
//            if (x2 >= 0 || x2 <= 1)x2 -= 1;
//            CubicSolver::generate(x0, x1, x2, as.data());
//            CubicSolver::dataType roots[3];
//            int numRoots = CubicSolver::solveCubicRange01(as[0], as[1], as[2], as[3], roots);
//            CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0]);
//            Catch::Approx(roots[0]) == x0;
//            INFO("Checking number of roots");
//            REQUIRE(numRoots == 1);
//            INFO("Checking the range of the root");
//            REQUIRE(roots[0] >= 0);
//            REQUIRE(roots[0] <= 1);
//        }
//    }
//
//    SECTION("Two roots within [0, 1]") {
//        for (int i = 0; i < CubicSolver::testNum; ++i) {
//            CubicSolver::dataType x0 = dist(rng);
//            CubicSolver::dataType x1 = dist(rng);
//            CubicSolver::dataType x2 = distBig(rng);
//            if (x2 >= 0 || x2 <= 1)x2 -= 1;
//            CubicSolver::generate(x0, x1, x2, as.data());
//            CubicSolver::dataType roots[3];
//            int numRoots = CubicSolver::solveCubicRange01(as[0], as[1], as[2], as[3], roots);
//            Catch::Approx(roots[0]) == x0;
//            Catch::Approx(roots[1]) == x1;
//            CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0], roots[1]);
//            INFO("Checking number of roots");
//            REQUIRE(numRoots == 2);
//            INFO("Checking the range of the root");
//            REQUIRE(roots[0] >= 0);
//            REQUIRE(roots[0] <= 1);
//            REQUIRE(roots[1] >= 0);
//            REQUIRE(roots[1] <= 1);
//            bool out01 = roots[2] < 0 || roots[2] > 1;
//            REQUIRE(out01);
//        }
//    }
//
//    SECTION("Three roots within [0, 1]") {
//        for (int i = 0; i < CubicSolver::testNum; ++i) {
//            CubicSolver::dataType x0 = dist(rng);
//            CubicSolver::dataType x1 = dist(rng);
//            CubicSolver::dataType x2 = dist(rng);
//            CubicSolver::generate(x0, x1, x2, as.data());
//            CubicSolver::dataType roots[3];
//            int numRoots = CubicSolver::solveCubicRange01(as[0], as[1], as[2], as[3], roots);
//            Catch::Approx(roots[0]) == x0;
//            Catch::Approx(roots[1]) == x1;
//            Catch::Approx(roots[2]) == x2;
//            CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0], roots[1], roots[2]);
//            INFO("Checking number of roots");
//            REQUIRE(numRoots == 3);
//            INFO("Checking the range of the root");
//            REQUIRE(roots[0] >= 0);
//            REQUIRE(roots[0] <= 1);
//            REQUIRE(roots[1] >= 0);
//            REQUIRE(roots[1] <= 1);
//            REQUIRE(roots[2] >= 0);
//            REQUIRE(roots[2] <= 1);
//        }
//    }
//
//}

TEST_CASE("tetrahedronTrajIntersection Tests", "[tetrahedronTrajIntersection]") {
    glm::vec3 X0;
    glm::vec3 V0;

    glm::vec3 x0;
    glm::vec3 x1;
    glm::vec3 x2;
    glm::vec3 x3;

    glm::vec3 XTilt0;
    glm::vec3 xTilt0;
    glm::vec3 xTilt1;
    glm::vec3 xTilt2;
    glm::vec3 xTilt3;
    X0 = glm::vec3(4.000000, 46.000000, -4.000000);
    V0 = glm::vec3(0.000011, -0.097931, -0.000010);
    XTilt0 = X0 + V0;
    x0 = glm::vec3(4.000000, 46.000000, 4.000000);
    x1 = glm::vec3(-4.000000, 46.000000, -4.000000);
    x2 = glm::vec3(-4.000000, 46.000000, 4.000000);
    x3 = glm::vec3(-4.000000, 54.000000, 4.000000);
    xTilt0 = glm::vec3(4.000014, 45.902069, 3.999993);
    xTilt1 = glm::vec3(-3.999996, 45.902061, -3.999997);
    xTilt2 = glm::vec3(-3.999984, 45.902058, 3.999999);
    xTilt3 = glm::vec3(-3.999989, 53.902054, 3.999998);
    const glm::vec3 v0 = xTilt0 - x0;
    const glm::vec3 v1 = xTilt1 - x1;
    const glm::vec3 v2 = xTilt2 - x2;
    const glm::vec3 v3 = xTilt3 - x3;
    SECTION("t equal 1.f") {
        const glm::vec3 v0 = xTilt0 - x0;
        const glm::vec3 v1 = xTilt1 - x1;
        const glm::vec3 v2 = xTilt2 - x2;
        const glm::vec3 v3 = xTilt3 - x3;
        dataType t = 1.f;
        // pre-test to check if the point is in the same plane as the triangle
            // check if the trajectory intersects with the triangle formed by the first three vertices
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x1, x2, v0, v1, v2));
        assert(t == 1.f);
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x1, x3, v0, v1, v3));
        assert(t == 1.f);
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x0, x2, x3, v0, v2, v3));
        assert(t == 1.f);
        t = glm::min(t, ccdTriangleIntersectionTest(X0, V0, x1, x2, x3, v1, v2, v3));
        assert(t == 1.f);
        INFO("Checking value of t equal 1.f");
        CAPTURE(X0, x0, x1, x2);
        REQUIRE(t == 1.f);
    }
}