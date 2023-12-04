#pragma once

#include <bvh.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <cuda_runtime.h>
#include <utilities.cuh>

__constant__ dataType AABBThreshold = 0.01;

__device__ AABB computeTetTrajBBox(const glmVec3& v0, const glmVec3& v1, const glmVec3& v2, const glmVec3& v3,
    const glmVec3& v4, const glmVec3& v5, const glmVec3& v6, const glmVec3& v7)
{
    glmVec3 min, max;
    min.x = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    min.y = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    min.z = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);
    max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);

    return AABB{ min - AABBThreshold, max + AABBThreshold };
}

struct MinOp {
    __host__ __device__
        glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const {
        return glm::min(a, b);
    }
};

struct MaxOp {
    __host__ __device__
        glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const {
        return glm::max(a, b);
    }
};

AABB computeBoundingBox(const thrust::device_ptr<glm::vec3>& begin, const thrust::device_ptr<glm::vec3>& end) {
    glm::vec3 min = thrust::reduce(begin, end, glm::vec3(FLT_MAX), MinOp());
    glm::vec3 max = thrust::reduce(begin, end, glm::vec3(-FLT_MAX), MaxOp());

    return AABB{ min, max };
}

AABB AABB::expand(const AABB& aabb)const {
    return AABB{
        glm::min(min, aabb.min),
        glm::max(max, aabb.max)
    };
}

CollisionDetection::CollisionDetection(const int _threadsPerBlock, size_t _maxQueries) :threadsPerBlock(_threadsPerBlock), maxNumQueries(_maxQueries)
{
    cudaMalloc(&dev_queries, maxNumQueries * sizeof(Query));

    cudaMalloc(&dev_numQueries, sizeof(size_t));
    cudaMemset(dev_numQueries, 0, sizeof(size_t));

    cudaMalloc(&dev_overflowFlag, sizeof(bool));
}

CollisionDetection::~CollisionDetection()
{
    cudaFree(dev_queries);
    cudaFree(dev_numQueries);
    cudaFree(dev_overflowFlag);
}

void CollisionDetection::DetectCollision(int numTets, const BVHNode* dev_BVHNodes, const GLuint* tets, const GLuint* tetFathers, const glm::vec3* Xs, const glm::vec3* XTilts, dataType*& tI, glm::vec3*& nors)
{
    if (BroadPhase(numTets, dev_BVHNodes, tets, tetFathers)) {
        PrepareRenderData(Xs);
        NarrowPhase(Xs, XTilts, tI, nors);
    }
}

BVH::BVH(const int _threadsPerBlock, size_t _maxQueries) : threadsPerBlock(_threadsPerBlock), collisionDetection(_threadsPerBlock, _maxQueries) {}

BVH::~BVH()
{
    cudaFree(dev_BVHNodes);
    cudaFree(dev_tI);
    cudaFree(dev_indicesToReport);

    cudaFree(dev_ready);
    cudaFree(dev_mortonCodes);
}

void BVH::PrepareRenderData()
{
    glm::vec3* pos;
    Wireframe::mapDevicePosPtr(&pos);
    dim3 numThreadsPerBlock(numNodes / threadsPerBlock + 1);
    populateBVHNodeAABBPos << <numThreadsPerBlock, threadsPerBlock >> > (dev_BVHNodes, pos, numNodes);
    Wireframe::unMapDevicePtr();
}

void BVH::DetectCollision(const GLuint* tets, const GLuint* tetFathers, const glm::vec3* Xs, const glm::vec3* XTilts, dataType* tI, glm::vec3* nors)
{
    thrust::device_ptr<dataType> dev_ptr(tI);
    thrust::fill(dev_ptr, dev_ptr + numVerts, 1.0f);
    collisionDetection.DetectCollision(numTets, dev_BVHNodes, tets, tetFathers, Xs, XTilts, tI, nors);
}

Drawable& BVH::GetQueryDrawable()
{
    return collisionDetection;
}

int BVH::GetNumQueries() const {
    return collisionDetection.GetNumQueries();
}
