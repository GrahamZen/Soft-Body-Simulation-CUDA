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

CollisionDetection::CollisionDetection(const SimulationCUDAContext* simContext, const int _threadsPerBlock, size_t _maxNumQueries) : mPSimContext(simContext), threadsPerBlock(_threadsPerBlock), maxNumQueries(_maxNumQueries), m_bvh(_threadsPerBlock)
{
    cudaMalloc(&dev_queries, maxNumQueries * sizeof(Query));

    cudaMalloc(&dev_numQueries, sizeof(size_t));
    cudaMemset(dev_numQueries, 0, sizeof(size_t));

    cudaMalloc(&dev_overflowFlag, sizeof(bool));
    mSqDisplay.create();
}

CollisionDetection::~CollisionDetection()
{
    cudaFree(dev_queries);
    cudaFree(dev_numQueries);
    cudaFree(dev_overflowFlag);
}

__global__ void processQueries(const Query* queries, int numQueries, glm::vec4* color) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numQueries) {
        Query q = queries[idx];
        atomicAdd(&color[q.v0].x, 0.05);
        atomicAdd(&color[q.v0].y, 0.05);
        atomicExch(&color[q.v0].w, 1);
    }
}

void CollisionDetection::PrepareRenderData(const glm::vec3* Xs)
{
    glm::vec3* pos;
    glm::vec4* col;
    MapDevicePosPtr(&pos, &col);
    cudaMemcpy(pos, Xs, numVerts * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemset(col, 0, numVerts * sizeof(glm::vec4));
    dim3 numBlocks((numQueries + threadsPerBlock - 1) / threadsPerBlock);
    processQueries << <numBlocks, threadsPerBlock >> > (dev_queries, numQueries, col);
    unMapDevicePtr();
}

BVH& CollisionDetection::GetBVH()
{
    return m_bvh;
}

SingleQueryDisplay& CollisionDetection::GetSQDisplay(int i, const glm::vec3* X, Query* guiQuery)
{
    if (numQueries == 0) {
        mSqDisplay.SetCount(0);
        return mSqDisplay;
    }
    mSqDisplay.SetCount(6);
    Query q;
    cudaMemcpy(&q, &dev_queries[i], sizeof(Query), cudaMemcpyDeviceToHost);
    if (guiQuery)
        *guiQuery = q;
    if (q.type == QueryType::EE) mSqDisplay.SetIsLine(true);
    else mSqDisplay.SetIsLine(false);
    if (mSqDisplay.IsLine()) {
        glm::vec3* pos;
        mSqDisplay.MapDevicePtr(&pos, nullptr, nullptr);
        cudaMemcpy(pos, &X[q.v0], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pos + 1, &X[q.v1], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pos + 2, &X[q.v2], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pos + 3, &X[q.v3], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        glm::vec3 v0Pos, v1Pos;
        cudaMemcpy(&v0Pos, &X[q.v0], sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v1Pos, &X[q.v1], sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pos[4], &((v0Pos + v1Pos) / 2.f), sizeof(glm::vec3), cudaMemcpyHostToDevice);
        // the third line point from the middle of v0 and v1 towards the normal direction
        glm::vec3 normalPoint = (v0Pos + v1Pos) / 2.f + q.normal * 10.f;
        cudaMemcpy(&pos[5], &normalPoint, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        mSqDisplay.UnMapDevicePtr();
    }
    else {
        glm::vec3* pos, * vertPos, * triPos;
        mSqDisplay.MapDevicePtr(&pos, &vertPos, &triPos);
        cudaMemcpy(vertPos, &X[q.v0], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(triPos, &X[q.v1], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(triPos + 1, &X[q.v2], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaMemcpy(triPos + 2, &X[q.v3], sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        glm::vec3 v0Pos;
        cudaMemcpy(&v0Pos, &X[q.v0], sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        glm::vec3 normalPoint = v0Pos + q.normal * 10.f;
        cudaMemcpy(&pos[0], &v0Pos, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(&pos[1], &normalPoint, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        mSqDisplay.UnMapDevicePtr();
    }
    return mSqDisplay;
}
