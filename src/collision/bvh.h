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
    void Init(int numTets, int numSoftBodies, int numVerts);
    void BuildBVHTree(int startIndexBVH, const AABB& ctxAABB, int triCount, const std::vector<SoftBody*>& softBodies);
    float* detectCollisionCandidates(GLuint* Tet, glm::vec3* X, glm::vec3* XTilt) const;
private:
    BVHNode* dev_BVHNodes = nullptr;
    AABB* dev_bboxes = nullptr;
    int numNodes;
    int numTets;
    int numVerts;
    float* dev_tI;
    int* dev_indicesToReport;
};