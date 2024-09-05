#pragma once

#include <collision/aabb.h>
#include <openglcontext/wireframe.h>
#include <openglcontext/queryDisplay.h>
#include <openglcontext/singleQueryDisplay.h>

class SoftBody;
class Context;
class SurfaceShader;

template<typename Scalar>
class BVH : public Wireframe {
public:
    enum class BuildType {
        Serial, Atomic, Cooperative
    };
    using ReadyFlagType = int;
    BVH<Scalar>(const int threadsPerBlock);
    ~BVH<Scalar>();
    void Init(int numTris, int numVerts, int maxThreads);
    void PrepareRenderData();
    const BVHNode<Scalar>* GetBVHNodes() const;
    void BuildBVHTree(BuildType buildType, const AABB<Scalar>& ctxAABB, int numTris, const glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, const indexType* tets);
private:
    void BuildBBoxes(BuildType buildType);
    BVHNode<Scalar>* dev_BVHNodes = nullptr;
    AABB<Scalar>* dev_bboxes = nullptr;
    unsigned int* dev_mortonCodes = nullptr;
    ReadyFlagType* dev_ready = nullptr;
    int numTris;
    Scalar* dev_tI;
    int* dev_indicesToReport;
    const int threadsPerBlock;
    dim3 numblocksTets;
    dim3 numblocksVerts;
    dim3 suggestedCGNumblocks;
    int suggestedBlocksize;
    bool isBuildBBCG = false;
};

template<typename Scalar>
class CollisionDetection : public QueryDisplay {
public:
    CollisionDetection<Scalar>(const SolverData<Scalar>* solverData, const Context* context, const int threadsPerBlock, size_t maxNumQueries);
    ~CollisionDetection<Scalar>();
    void DetectCollision(Scalar* tI, glm::vec3* nors);
    void Init(int numTris, int numVerts, int maxThreads);
    void PrepareRenderData();
    void Draw(SurfaceShader*);
    int GetNumQueries() const {
        return numQueries;
    }
    void SetBuildType(typename BVH<Scalar>::BuildType);
    typename BVH<Scalar>::BuildType GetBuildType();
private:
    SingleQueryDisplay& GetSQDisplay(int i, const glm::tvec3<Scalar>* Xs, Query* guiQuery);
    AABB<Scalar> GetAABB() const;
    bool BroadPhase();
    void NarrowPhase(Scalar*& tI, glm::vec3*& nors);
    bool DetectCollisionCandidates(const BVHNode<Scalar>* dev_BVHNodes);
    Query* dev_queries;
    SingleQueryDisplay mSqDisplay;
    size_t* dev_numQueries;
    size_t numQueries;
    size_t maxNumQueries = 1 << 15;
    bool* dev_overflowFlag;
    const int threadsPerBlock;
    const SolverData<Scalar>* mpSolverData;
    const Context* mpContext;
    BVH<Scalar> m_bvh;
    typename BVH<Scalar>::BuildType buildType = BVH<Scalar>::BuildType::Atomic;
};