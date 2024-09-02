#pragma once

#include <collision/aabb.h>
#include <openglcontext/wireframe.h>
#include <openglcontext/queryDisplay.h>
#include <openglcontext/singleQueryDisplay.h>

class SoftBody;
class Context;
class SurfaceShader;

template<typename HighP>
class BVH : public Wireframe {
public:
    enum class BuildType {
        Serial, Atomic, Cooperative
    };
    using ReadyFlagType = int;
    BVH<HighP>(const int threadsPerBlock);
    ~BVH<HighP>();
    void Init(int numTets, int numVerts, int maxThreads);
    void PrepareRenderData();
    const BVHNode<HighP>* GetBVHNodes() const;
    void BuildBVHTree(BuildType buildType, const AABB<HighP>& ctxAABB, int numTets, const glm::tvec3<HighP>* X, glm::tvec3<HighP>* XTilde, const indexType* tets);
private:
    void BuildBBoxes(BuildType buildType);
    BVHNode<HighP>* dev_BVHNodes = nullptr;
    AABB<HighP>* dev_bboxes = nullptr;
    unsigned int* dev_mortonCodes = nullptr;
    ReadyFlagType* dev_ready = nullptr;
    int numTets;
    HighP* dev_tI;
    int* dev_indicesToReport;
    const int threadsPerBlock;
    dim3 numblocksTets;
    dim3 numblocksVerts;
    dim3 suggestedCGNumblocks;
    int suggestedBlocksize;
    bool isBuildBBCG = false;
};

template<typename HighP>
class CollisionDetection : public QueryDisplay {
public:
    CollisionDetection<HighP>(const SolverData<HighP>* solverData, const Context* context, const int threadsPerBlock, size_t maxNumQueries);
    ~CollisionDetection<HighP>();
    void DetectCollision(HighP* tI, glm::vec3* nors);
    void Init(int numTets, int numVerts, int maxThreads);
    void PrepareRenderData();
    void Draw(SurfaceShader*);
    int GetNumQueries() const {
        return numQueries;
    }
    void SetBuildType(typename BVH<HighP>::BuildType);
    typename BVH<HighP>::BuildType GetBuildType();
private:
    SingleQueryDisplay& GetSQDisplay(int i, const glm::tvec3<HighP>* Xs, Query* guiQuery);
    AABB<HighP> GetAABB() const;
    bool BroadPhase();
    void NarrowPhase(HighP*& tI, glm::vec3*& nors);
    bool DetectCollisionCandidates(const BVHNode<HighP>* dev_BVHNodes);
    Query* dev_queries;
    SingleQueryDisplay mSqDisplay;
    size_t* dev_numQueries;
    size_t numQueries;
    size_t maxNumQueries = 1 << 15;
    bool* dev_overflowFlag;
    const int threadsPerBlock;
    const SolverData<HighP>* mpSolverData;
    const Context* mpContext;
    BVH<HighP> m_bvh;
    typename BVH<HighP>::BuildType buildType = BVH<HighP>::BuildType::Atomic;
};