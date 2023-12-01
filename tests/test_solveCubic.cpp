#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>
#include <iostream>

namespace CubicSolver {
    using dataType = double;
    int testNum = 1000;
    template<typename T>
    int solveQuadratic(T a, T b, T c, T& t0, T& t1) {
        T d = b * b - 4.0f * a * c;
        if (d < 0.0f) {
            return 0;
        }
        T eps = glm::epsilon<T>();
        int result_num = 0;
        T q = -0.5f * (b + glm::sign(b) * glm::sqrt(d));
        if (glm::abs(a) > eps * glm::abs(q)) {
            t0 = q / a;
            result_num += 1;
        }
        if (glm::abs(q) > eps * glm::abs(c)) {
            t1 = c / q;
            result_num += 1;
        }
        if (result_num == 2 && t0 > t1) {
            T temp = t0;
            t0 = t1;
            t1 = temp;
        }
        return result_num;
    }
    const float M_PI = glm::pi<float>();
    template<typename T>
    int solve_quadratic(T a, T b, T c, T* x) {
        // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
        T d = b * b - 4 * a * c;
        if (d < 0) {
            x[0] = -b / (2 * a);
            return 0;
        }
        T q = -(b + glm::sign(b) * sqrt(d)) / 2;
        int i = 0;
        if (abs(a) > 1e-12 * abs(q))
            x[i++] = q / a;
        if (abs(q) > 1e-12 * abs(c))
            x[i++] = c / q;
        if (i == 2 && x[0] > x[1]) {
            T tmp = x[0];
            x[0] = x[1];
            x[1] = tmp;
        }
        return i;
    }
    template<typename T>
    int solveCubic(T a, T b, T c, T d, T* x) {
        T xc[2];
        int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
        if (ncrit == 0) {
            x[0] = newtons_method(a, b, c, d, xc[0], 0);
            return 1;
        }
        else if (ncrit == 1) {// cubic is actually quadratic
            return solve_quadratic(b, c, d, x);
        }
        else {
            T yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
                            d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
            int i = 0;
            if (yc[0] * a >= 0)
                x[i++] = newtons_method(a, b, c, d, xc[0], -1);
            if (yc[0] * yc[1] <= 0) {
                int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
                x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
            }
            if (yc[1] * a <= 0)
                x[i++] = newtons_method(a, b, c, d, xc[1], 1);
            return i;
        }
    }

    template<typename T>
    T newtons_method(T a, T b, T c, T d, T x0,
        int init_dir) {
        if (init_dir != 0) {
            // quadratic approximation around x0, assuming y' = 0
            T y0 = d + x0 * (c + x0 * (b + x0 * a)),
                ddy0 = 2 * b + x0 * (6 * a);
            x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
        }
        for (int iter = 0; iter < 100; iter++) {
            T y = d + x0 * (c + x0 * (b + x0 * a));
            T dy = c + x0 * (2 * b + x0 * 3 * a);
            if (dy == 0)
                return x0;
            T x1 = x0 - y / dy;
            if (abs(x0 - x1) < 1e-6)
                return x0;
            x0 = x1;
        }
        return x0;
    }
    template<typename T>
    T solveCubicRange01(T a, T b, T c, T d, T* x) {
        T roots[3];
        int j = 0;
        int numRoots = solveCubic(a, b, c, d, roots);
        for (int i = 0; i < numRoots; i++) {
            if (roots[i] >= 0 && roots[i] <= 1) {
                x[j++] = roots[i];
            }
        }
        return j;
    }
    template<typename T>
    T solveCubicMinGtZero(T a, T b, T c, T d) {
        T t[3];
        int t_num = solveCubic(a, b, c, d, t);
        T min_t = FLT_MAX;
        for (int i = 0; i < t_num; i++) {
            if (t[i] > 0.0f && t[i] < 1.0f) {
                min_t = glm::min(min_t, t[i]);
            }
        }
        if (min_t == FLT_MAX) {
            return -1.0f;
        }
        return min_t;
    }
    template<typename T>
    void generate(T x0, T x1, T x2, T* a) {
        a[0] = 1;
        a[1] = -(x0 + x1 + x2);
        a[2] = x0 * x1 + x1 * x2 + x2 * x0;
        a[3] = -x0 * x1 * x2;
    }
};

TEST_CASE("solveCubic Tests", "[solveCubic]") {
    std::mt19937 rng;
    std::uniform_real_distribution<CubicSolver::dataType> dist(0.0, 1.0);
    std::uniform_real_distribution<CubicSolver::dataType> distBig(-9999, 9999);
    std::vector<CubicSolver::dataType>as(4);
    SECTION("No roots within [0, 1]") {
        CubicSolver::dataType a = 1, b = 0, c = 0, d = 1;
        CubicSolver::dataType roots[3];
        int numRoots = CubicSolver::solveCubicRange01(a, b, c, d, roots);
        INFO("Checking number of roots");
        REQUIRE(numRoots == 0);
    }

    SECTION("One root within [0, 1]") {
        for (int i = 0; i < CubicSolver::testNum; ++i) {
            CubicSolver::dataType x0 = dist(rng);
            CubicSolver::dataType x1 = distBig(rng);
            if (x1 >= 0 || x1 <= 1)x1 -= 1;
            CubicSolver::dataType x2 = distBig(rng);
            if (x2 >= 0 || x2 <= 1)x2 -= 1;
            CubicSolver::generate(x0, x1, x2, as.data());
            CubicSolver::dataType roots[3];
            int numRoots = CubicSolver::solveCubicRange01(as[0], as[1], as[2], as[3], roots);
            CAPTURE(x0, x1, x2, as[0], as[1], as[2], roots[0]);
            Catch::Approx(roots[0]) == x0;
            INFO("Checking number of roots");
            REQUIRE(numRoots == 1);
            INFO("Checking the range of the root");
            REQUIRE(roots[0] >= 0);
            REQUIRE(roots[0] <= 1);
        }
    }

    SECTION("Two roots within [0, 1]") {
        for (int i = 0; i < CubicSolver::testNum; ++i) {
            CubicSolver::dataType x0 = dist(rng);
            CubicSolver::dataType x1 = dist(rng);
            CubicSolver::dataType x2 = distBig(rng);
            if (x2 >= 0 || x2 <= 1)x2 -= 1;
            CubicSolver::generate(x0, x1, x2, as.data());
            CubicSolver::dataType roots[3];
            int numRoots = CubicSolver::solveCubicRange01(as[0], as[1], as[2], as[3], roots);
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

    SECTION("Three roots within [0, 1]") {
        for (int i = 0; i < CubicSolver::testNum; ++i) {
            CubicSolver::dataType x0 = dist(rng);
            CubicSolver::dataType x1 = dist(rng);
            CubicSolver::dataType x2 = dist(rng);
            CubicSolver::generate(x0, x1, x2, as.data());
            CubicSolver::dataType roots[3];
            int numRoots = CubicSolver::solveCubicRange01(as[0], as[1], as[2], as[3], roots);
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
