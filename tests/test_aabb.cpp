#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>
#include <intersections.h>
#include <iostream>

TEST_CASE("AABB Tests", "[AABB]") {
    glm::vec3 x0;
    glm::vec3 xTilt;
    AABB bbox;
    x0 = glm::vec3(3.990862, 4.531781, -3.997540), xTilt = glm::vec3(4.005919, 1.919091, -3.993216), bbox = AABB{ glm::vec3(-3.995214, -4.082155, -4.012133), glm::vec3(4.005711, 3.912421, 3.997659) };
    bool res = edgeBboxIntersectionTest(x0, xTilt, bbox);
    REQUIRE(res == true);
}
