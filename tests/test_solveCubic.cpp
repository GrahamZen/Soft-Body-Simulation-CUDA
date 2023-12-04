#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/intersect.hpp>
#include <intersections.h>
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

}
