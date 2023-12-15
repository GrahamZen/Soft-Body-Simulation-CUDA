#pragma once

#include <def.h>
#include <wireframe.h>
#include <mesh.h>
#include <queryDisplay.h>
#include <singleQueryDisplay.h>

class SoftBody;
class SimulationCUDAContext;
class SurfaceShader;
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
    indexType v0;
    indexType v1;
    indexType v2;
    indexType v3;
    float toi = 0.f;
    glm::vec3 normal = glm::vec3(0.f);
};

class BVH : public Wireframe {
public:
    enum class BuildType {
        Serial, Atomic, Cooperative
    };
    using ReadyFlagType = int;
    BVH(const int threadsPerBlock);
    ~BVH();
    void Init(int numTets, int numVerts, int maxThreads);
    void PrepareRenderData();
    const BVHNode* GetBVHNodes() const;
    void BuildBVHTree(BuildType buildType, const AABB& ctxAABB, int numTets, const glm::vec3* X, const glm::vec3* XTilt, const indexType* tets);
private:
    void BuildBBoxes(BuildType buildType);
    BVHNode* dev_BVHNodes = nullptr;
    AABB* dev_bboxes = nullptr;
    unsigned int* dev_mortonCodes = nullptr;
    ReadyFlagType* dev_ready = nullptr;
    int numTets;
    dataType* dev_tI;
    int* dev_indicesToReport;
    const int threadsPerBlock;
    dim3 numblocksTets;
    dim3 numblocksVerts;
    dim3 suggestedCGNumblocks;
    int suggestedBlocksize;
    bool isBuildBBCG = false;
};

class CollisionDetection : public QueryDisplay {
public:
    CollisionDetection(const SimulationCUDAContext* simContext, const int threadsPerBlock, size_t maxNumQueries);
    ~CollisionDetection();
    void DetectCollision(dataType* tI, glm::vec3* nors);
    void Init(int numTets, int numVerts, int maxThreads);
    void PrepareRenderData();
    void Draw(SurfaceShader*);
    int GetNumQueries() const {
        return numQueries;
    }
    void SetBuildType(BVH::BuildType);
    BVH::BuildType GetBuildType();
private:
    SingleQueryDisplay& GetSQDisplay(int i, const glm::vec3* Xs, Query* guiQuery);
    AABB GetAABB() const;
    bool BroadPhase();
    void NarrowPhase(dataType*& tI, glm::vec3*& nors);
    bool DetectCollisionCandidates(const BVHNode* dev_BVHNodes);
    Query* dev_queries;
    SingleQueryDisplay mSqDisplay;
    size_t* dev_numQueries;
    size_t numQueries;
    size_t maxNumQueries = 1 << 15;
    bool* dev_overflowFlag;
    const int threadsPerBlock;
    const SimulationCUDAContext* mPSimContext;
    BVH m_bvh;
    BVH::BuildType buildType = BVH::BuildType::Atomic;
};