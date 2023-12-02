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
    int numTets;
    std::vector<GLuint> tets = { 3, 1, 0, 2, 7, 5, 4, 6 };
    glmVec3 X0 = glmVec3(0.0f, 0.0f, 0.0f);
    glmVec3 XTilt = glmVec3(0.0f, 0.0f, 0.0f);
    std::vector<glm::vec3> Xs;
    std::vector<glm::vec3> XTilts;
    GLuint tetId;

    Xs = std::vector<glm::vec3>{
        glm::vec3(3.983533, 15.103371, -3.986765),
        glm::vec3(-3.995983, 7.082971, -4.001955),
        glm::vec3(4.002815, 7.908820, -4.002882),
        glm::vec3(4.007270, 7.085022, 3.994780),
        glm::vec3(4.008490, 7.906959, -3.995315),
        glm::vec3(-3.991498, 0.000000, -3.998166),
        glm::vec3(4.008802, 0.000000, -3.996651),
        glm::vec3(4.006739, 0.000000, 4.003596),
    };
    XTilts = std::vector<glm::vec3>{
        glm::vec3(3.937732, 12.668071, -4.263350),
        glm::vec3(-4.061966, 4.686636, -3.735191),
        glm::vec3(3.927908, 4.681063, -3.993726),
        glm::vec3(4.193937, 4.940782, 3.995438),
        glm::vec3(4.005813, 7.904163, -3.995533),
        glm::vec3(-3.992768, 0.000000, -3.993595),
        glm::vec3(4.007134, 0.000000, -3.995270),
        glm::vec3(4.008984, 0.000000, 4.004651) };
    int vertIdx = 2;
    X0 = Xs[vertIdx];
    XTilt = XTilts[vertIdx];
    tetId = 1;
    std::vector<glmVec3> x{ Xs[tets[tetId * 4 + 0]], Xs[tets[tetId * 4 + 1]], Xs[tets[tetId * 4 + 2]], Xs[tets[tetId * 4 + 3]] },
        xTilt{ XTilts[tets[tetId * 4 + 0]], XTilts[tets[tetId * 4 + 1]], XTilts[tets[tetId * 4 + 2]], XTilts[tets[tetId * 4 + 3]] };
    std::vector<int> indices{ 0, 1, 2 };
    std::cout << "x1 = " << x[indices[0]].x
        << std::endl
        << "y1 = " << x[indices[0]].y
        << std::endl
        << "z1 = " << x[indices[0]].z
        << std::endl
        << "x2 = " << x[indices[1]].x
        << std::endl
        << "y2 = " << x[indices[1]].y
        << std::endl
        << "z2 = " << x[indices[1]].z
        << std::endl
        << "x3 = " << x[indices[2]].x
        << std::endl
        << "y3 = " << x[indices[2]].y
        << std::endl
        << "z3 = " << x[indices[2]].z
        << std::endl
        << "x4 = " << xTilt[indices[0]].x
        << std::endl
        << "y4 = " << xTilt[indices[0]].y
        << std::endl
        << "z4 = " << xTilt[indices[0]].z
        << std::endl
        << "x5 = " << xTilt[indices[1]].x
        << std::endl
        << "y5 = " << xTilt[indices[1]].y
        << std::endl
        << "z5 = " << xTilt[indices[1]].z
        << std::endl
        << "x6 = " << xTilt[indices[2]].x
        << std::endl
        << "y6 = " << xTilt[indices[2]].y
        << std::endl
        << "z6 = " << xTilt[indices[2]].z
        << std::endl
        << "lx1 = " << X0.x
        << std::endl
        << "ly1 = " << X0.y
        << std::endl
        << "lz1 = " << X0.z
        << std::endl
        << "lx2 = " << XTilt.x
        << std::endl
        << "ly2 = " << XTilt.y
        << std::endl
        << "lz2 = " << XTilt.z << std::endl;

    dataType t = tetrahedronTrajIntersectionTest(tets.data(), X0, XTilt, Xs.data(), XTilts.data(), tetId);
    REQUIRE(t < 1.0f);
}
