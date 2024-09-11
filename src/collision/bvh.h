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
    void BuildBVHTreeCCD(BuildType buildType, const AABB<Scalar>& ctxAABB, int numTris, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, const indexType* tets);
    void BuildBVHTree(BuildType buildType, const AABB<Scalar>& ctxAABB, int numTris, const glm::tvec3<Scalar>* X, const indexType* tets);
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
    CollisionDetection<Scalar>(const Context* context, const int threadsPerBlock, size_t maxNumQueries);
    ~CollisionDetection<Scalar>();
    void DetectCollision(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde,
        const indexType* TriFathers, Scalar* tI, glm::vec3* nors, bool ignoreSelfCollision = false);
    Scalar ComputeMinStepSize(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde,
        const indexType* TriFathers, bool ignoreSelfCollision = false);
    bool BroadPhase(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const indexType* TriFathers, Query*& queries, int& _numQueries);
    void Init(int numTris, int numVerts, int maxThreads);
    void PrepareRenderData();
    void Draw(SurfaceShader*);
    int GetNumQueries() const {
        return numQueries;
    }
    void SetBuildType(int);
    int GetBuildType();
private:
    SingleQueryDisplay& GetSQDisplay(int i, const glm::tvec3<Scalar>* Xs, Query* guiQuery);
    bool BroadPhaseCCD(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, const indexType* TriFathers);
    void NarrowPhase(const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, Scalar*& tI, glm::vec3*& nors);
    Scalar NarrowPhase(const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde);
    bool DetectCollisionCandidates(const BVHNode<Scalar>* dev_BVHNodes, int numTris, const indexType* Tri, const indexType* TriFathers);
    Query* dev_queries;
    SingleQueryDisplay mSqDisplay;
    size_t* dev_numQueries;
    size_t numQueries;
    size_t maxNumQueries = 1 << 15;
    bool* dev_overflowFlag;
    bool ignoreSelfCollision = false;
    const int threadsPerBlock;
    const Context* mpContext;
    const glm::tvec3<Scalar>* mpX;
    BVH<Scalar> m_bvh;
    typename BVH<Scalar>::BuildType buildType = BVH<Scalar>::BuildType::Atomic;
};