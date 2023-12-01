#pragma once

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <wireframe.h>

class SoftBody;

using dataType = double;
using glmVec4 = glm::tvec4<dataType>;
using glmVec3 = glm::tvec3<dataType>;
using glmVec2 = glm::tvec2<dataType>;
using glmMat4 = glm::tmat4x4<dataType>;
using glmMat3 = glm::tmat3x3<dataType>;
using glmMat2 = glm::tmat2x2<dataType>;

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

    BVH(const int threadsPerBlock);
    ~BVH();
    void Init(int numTets, int numVerts, int maxThreads);
    void BuildBVHTree(const AABB& ctxAABB, int numTets, const glm::vec3* X, const glm::vec3* XTilt, const GLuint* tets);
    dataType* DetectCollisionCandidates(const GLuint* edges, const GLuint* tets, const glm::vec3* Xs, const glm::vec3* XTilts) const;
    void PrepareRenderData();
private:
    void BuildBBoxes();
    BVHNode* dev_BVHNodes = nullptr;
    AABB* dev_bboxes = nullptr;
    unsigned int* dev_mortonCodes = nullptr;
    unsigned char* dev_ready = nullptr;

    int numNodes;
    int numTets;
    int numVerts;
    int numEdges;
    dataType* dev_tI;
    int* dev_indicesToReport;
    const int threadsPerBlock;
    dim3 numblocks;
    dim3 suggestedCGNumblocks;
    int suggestedBlocksize;
    bool isBuildBBCG = false;
};