#pragma once

#include <vector>
#include <glm/glm.hpp>

class SoftBody;

struct AABB {
    glm::vec3 min = glm::vec3{ FLT_MAX };
    glm::vec3 max = glm::vec3{ -FLT_MAX };
    AABB expand(const AABB& aabb)const;
};
struct BVHNode {
    AABB bbox;
    int isLeaf;
    int leftIndex;
    int rightIndex;
    int parent;
    int TetrahedronIndex;
};

class BVH {
public:
    BVH() = default;
    ~BVH();
    void Init(int, int);
    void BuildBVHTree(int startIndexBVH, const AABB& ctxAABB, int triCount, const std::vector<SoftBody*>& softBodies);
    std::vector<std::pair<int, int>> detectCollisionCandidates(GLuint* Tet, int numTet, glm::vec3* X, int number);
private:
    BVHNode* dev_BVHNodes = nullptr;
    AABB* dev_bboxes = nullptr;
    int numNodes;
};