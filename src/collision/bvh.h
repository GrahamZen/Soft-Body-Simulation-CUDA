#pragma once

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <wireframe.h>

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

class BVH : public Wireframe {
public:
    enum class BuildMethodType
    {
        SERIAL,
        PARALLEL,
        COOPERATIVE_GROUP
    };
    BVH(const int threadsPerBlock, BuildMethodType buildMethod = BuildMethodType::PARALLEL);
    ~BVH();
    void Init(int numTets, int numVerts);
    void BuildBVHTree(const AABB& ctxAABB, int numTets, const glm::vec3* X, const glm::vec3* XTilt, const GLuint* tets);
    float* DetectCollisionCandidates(const GLuint* Tet, const glm::vec3* Xs, const glm::vec3* XTilts, const GLuint* TetId) const;
    void PrepareRenderData();
private:
    BVHNode* dev_BVHNodes = nullptr;
    AABB* dev_bboxes = nullptr;
    unsigned int* dev_mortonCodes = nullptr;
    int* dev_ready = nullptr;

    int numNodes;
    int numTets;
    int numVerts;
    float* dev_tI;
    int* dev_indicesToReport;
    const int threadsPerBlock;
    BuildMethodType buildMethod;
};