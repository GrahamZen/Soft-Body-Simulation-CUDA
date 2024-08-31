#pragma once

#include <utilities.cuh>
#include <surfaceshader.h>
#include <collision/bvh.h>
#include <simulation/simulationContext.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

__constant__ colliPrecision AABBThreshold = 0.01;

template<typename HighP>
struct HighPtoFloatP {
    __host__ __device__ glm::vec3 operator()(const glm::tvec3<HighP>& d) {
        return glm::vec3(static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z));
    }
};

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

void CollisionDetection::PrepareRenderData()
{
    if (mPSimContext->context->guiData->QueryVis && numQueries > 0) {
        glm::vec3* pos;
        glm::vec4* col;
        MapDevicePosPtr(&pos, &col);
        thrust::device_ptr<const glm::tvec3<solverPrecision>> dvec3_ptr(mPSimContext->mSolverData.X);
        thrust::device_ptr<glm::vec3> vec3_ptr(pos);

        thrust::transform(dvec3_ptr, dvec3_ptr + numVerts, vec3_ptr,
            [] __host__ __device__(const glm::tvec3<solverPrecision> &d) {
            return glm::vec3(static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z));
        });
        cudaMemset(col, 0, numVerts * sizeof(glm::vec4));
        dim3 numBlocks((numQueries + threadsPerBlock - 1) / threadsPerBlock);
        processQueries << <numBlocks, threadsPerBlock >> > (dev_queries, numQueries, col);
        UnMapDevicePtr();
    }
    if (mPSimContext->context->guiData->BVHVis) {
        m_bvh.PrepareRenderData();
    }
}

void CollisionDetection::Draw(SurfaceShader* flatShaderProgram)
{
    if (mPSimContext->context->guiData->BVHVis)
        flatShaderProgram->draw(m_bvh, 0);
    if (mPSimContext->context->guiData->QueryVis)
        flatShaderProgram->drawPoints(*this);
    if (mPSimContext->context->guiData->QueryDebugMode) {
        glLineWidth(mPSimContext->context->guiData->LineWidth);
        flatShaderProgram->drawSingleQuery(GetSQDisplay(mPSimContext->context->guiData->CurrQueryId, mPSimContext->mSolverData.X,
            mPSimContext->context->guiData->QueryDirty ? mPSimContext->context->guiData->mPQuery : nullptr));
        mPSimContext->context->guiData->QueryDirty = false;
    }
}

template<typename HighP>
SingleQueryDisplay& CollisionDetection::GetSQDisplay(int i, const glm::tvec3<HighP>* X, Query* guiQuery)
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
        thrust::device_ptr<glm::vec3> dev_ptr(pos);
        thrust::device_ptr<const glm::tvec3<HighP>> dev_ptr_X(X);
        thrust::transform(dev_ptr_X + q.v0, dev_ptr_X + q.v0 + 1, dev_ptr, HighPtoFloatP<HighP>());
        thrust::transform(dev_ptr_X + q.v1, dev_ptr_X + q.v1 + 1, dev_ptr + 1, HighPtoFloatP<HighP>());
        thrust::transform(dev_ptr_X + q.v2, dev_ptr_X + q.v2 + 1, dev_ptr + 2, HighPtoFloatP<HighP>());
        thrust::transform(dev_ptr_X + q.v3, dev_ptr_X + q.v3 + 1, dev_ptr + 3, HighPtoFloatP<HighP>());

        glm::vec3 v0Pos, v1Pos;
        cudaMemcpy(&v0Pos, pos + 1, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v1Pos, pos + 2, sizeof(glm::vec3), cudaMemcpyDeviceToHost);

        cudaMemcpy(&pos[4], &((v0Pos + v1Pos) / 2.f), sizeof(glm::vec3), cudaMemcpyHostToDevice);
        // the third line point from the middle of v0 and v1 towards the normal direction
        glm::vec3 normalPoint = (v0Pos + v1Pos) / 2.f + q.normal * 10.f;
        cudaMemcpy(&pos[5], &normalPoint, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        mSqDisplay.UnMapDevicePtr(&pos, nullptr, nullptr);
    }
    else {
        glm::vec3* pos, * vertPos, * triPos;
        mSqDisplay.MapDevicePtr(&pos, &vertPos, &triPos);
        thrust::device_ptr<glm::vec3> dev_vertPos(vertPos);
        thrust::device_ptr<glm::vec3> dev_triPos(triPos);
        thrust::device_ptr<const glm::tvec3<HighP>> dev_ptr_X(X);
        thrust::transform(dev_ptr_X + q.v0, dev_ptr_X + q.v0 + 1, dev_vertPos, HighPtoFloatP<HighP>());
        thrust::transform(dev_ptr_X + q.v1, dev_ptr_X + q.v1 + 1, dev_triPos, HighPtoFloatP<HighP>());
        thrust::transform(dev_ptr_X + q.v2, dev_ptr_X + q.v2 + 1, dev_triPos + 1, HighPtoFloatP<HighP>());
        thrust::transform(dev_ptr_X + q.v3, dev_ptr_X + q.v3 + 1, dev_triPos + 2, HighPtoFloatP<HighP>());
        glm::vec3 v0Pos;
        cudaMemcpy(&v0Pos, vertPos, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        glm::vec3 normalPoint = v0Pos + q.normal * 10.f;
        cudaMemcpy(&pos[0], &v0Pos, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(&pos[1], &normalPoint, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        mSqDisplay.UnMapDevicePtr(&pos, &vertPos, &triPos);
    }
    return mSqDisplay;
}
