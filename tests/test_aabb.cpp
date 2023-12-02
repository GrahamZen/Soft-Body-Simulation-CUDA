#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/constants.hpp>
#include <intersections.h>
#include <iostream>

TEST_CASE("AABB Tests impossible", "[AABB]") {
    glm::vec3 x0, xTilt;
    AABB bbox;
    x0 = glmVec3(4.000087, 9.568024, -3.996036), xTilt = glmVec3(4.004138, 7.104599, -4.006201);
    bbox = AABB{ glmVec3(-3.995531, -0.089164, -3.999941), glmVec3(4.010788, 7.908782, 4.006889) };
    bool res = edgeBboxIntersectionTest(x0, xTilt, bbox);
    REQUIRE(res == false);
}