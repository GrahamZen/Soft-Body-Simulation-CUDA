#pragma once

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <wireframe.h>
#include <mesh.h>
#include <queryDisplay.h>

class SoftBody;

using dataType = double;
using glmVec4 = glm::tvec4<dataType>;
using glmVec3 = glm::tvec3<dataType>;
using glmVec2 = glm::tvec2<dataType>;
using glmMat4 = glm::tmat4x4<dataType>;
using glmMat3 = glm::tmat3x3<dataType>;
using glmMat2 = glm::tmat2x2<dataType>;

class AABB {
public:
    glm::vec3 min = glm::vec3{ FLT_MAX };
    glm::vec3 max = glm::vec3{ -FLT_MAX };
    AABB expand(const AABB& aabb)const;
};
class BVHNode {
public:
    AABB bbox;
    int isLeaf;
    int leftIndex;
    int rightIndex;
    int parent;
    int TetrahedronIndex;
};

enum class QueryType {
    UNKNOWN,
    VF,
    EE
};

class Query {
public:
    QueryType type;
    GLuint v0;
    GLuint v1;
    GLuint v2;
    GLuint v3;
    float toi = 0.f;
    glm::vec3 normal = glm::vec3(0.f);
};

class CollisionDetection : public QueryDisplay {
public:
    CollisionDetection(const int threadsPerBlock, size_t maxNumQueries);
    ~CollisionDetection();
    bool DetectCollisionCandidates(int numTets, const BVHNode* dev_BVHNodes, const GLuint* tets, const GLuint* tetFathers);
    bool BroadPhase(int numTets, const BVHNode* dev_BVHNodes, const GLuint* tets, const GLuint* tetFathers);
    void DetectCollision(int numTets, const BVHNode* dev_BVHNodes, const GLuint* tets, const GLuint* tetFathers, const glm::vec3* Xs, const glm::vec3* XTilts, dataType*& tI, glm::vec3*& nors, const glm::vec3* X0 = nullptr);
    void NarrowPhase(const glm::vec3* Xs, const glm::vec3* XTilts, dataType*& tI, glm::vec3*& nors);
    void PrepareRenderData(const glm::vec3* Xs);
    int GetNumQueries() const {
        return numQueries;
    }
private:
    Query* dev_queries;
    size_t* dev_numQueries;
    size_t numQueries;
    size_t maxNumQueries = 1 << 15;
    bool* dev_overflowFlag;
    const int threadsPerBlock;
};

class BVH : public Wireframe {
public:

    BVH(const int threadsPerBlock, size_t _maxQueries);
    ~BVH();
    void Init(int numTets, int numVerts, int maxThreads);
    void PrepareRenderData();
    void BuildBVHTree(const AABB& ctxAABB, int numTets, const glm::vec3* X, const glm::vec3* XTilt, const GLuint* tets);
    void DetectCollision(const GLuint* tets, const GLuint* tetFathers, const glm::vec3* Xs, const glm::vec3* XTilts, dataType* tI, glm::vec3* nors, const glm::vec3* X0 = nullptr);
    Drawable& GetQueryDrawable();
    int GetNumQueries() const;
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
    dim3 numblocksTets;
    dim3 numblocksVerts;
    dim3 suggestedCGNumblocks;
    int suggestedBlocksize;
    bool isBuildBBCG = false;

    CollisionDetection collisionDetection;
};