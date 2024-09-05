#pragma once

#include <collision/bvh.h>
#include <utilities.cuh>
#include <surfaceshader.h>
#include <context.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

__constant__ double AABBThreshold = 0.01;

template<typename Scalar>
struct HighPtoFloatP {
    __host__ __device__ glm::vec3 operator()(const glm::tvec3<Scalar>& d) {
        return glm::vec3(static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z));
    }
};

template<typename Scalar>
__device__ AABB<Scalar> computeTetTrajBBox(const glm::tvec3<Scalar>& v0, const glm::tvec3<Scalar>& v1, const glm::tvec3<Scalar>& v2, const glm::tvec3<Scalar>& v3,
    const glm::tvec3<Scalar>& v4, const glm::tvec3<Scalar>& v5, const glm::tvec3<Scalar>& v6, const glm::tvec3<Scalar>& v7)
{
    glm::tvec3<Scalar> min, max;
    min.x = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    min.y = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    min.z = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);
    max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);

    return AABB<Scalar>{ min - (Scalar)AABBThreshold, max + (Scalar)AABBThreshold };
}

template<typename Scalar>
__device__ AABB<Scalar> computeTriTrajBBox(const glm::tvec3<Scalar>& v0, const glm::tvec3<Scalar>& v1, const glm::tvec3<Scalar>& v2, const glm::tvec3<Scalar>& v3,
    const glm::tvec3<Scalar>& v4, const glm::tvec3<Scalar>& v5)
{
    glm::tvec3<Scalar> min, max;
    min.x = fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    min.y = fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    min.z = fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);
    max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);

    return AABB<Scalar>{ min - (Scalar)AABBThreshold, max + (Scalar)AABBThreshold };
}

template __device__ AABB<float> computeTriTrajBBox(const glm::tvec3<float>& v0, const glm::tvec3<float>& v1, const glm::tvec3<float>& v2, const glm::tvec3<float>& v3,
    const glm::tvec3<float>& v4, const glm::tvec3<float>& v5);

template __device__ AABB<double> computeTriTrajBBox(const glm::tvec3<double>& v0, const glm::tvec3<double>& v1, const glm::tvec3<double>& v2, const glm::tvec3<double>& v3,
    const glm::tvec3<double>& v4, const glm::tvec3<double>& v5);

template<typename Scalar>
AABB<Scalar> AABB<Scalar>::expand(const AABB<Scalar>& aabb)const {
    return AABB<Scalar>{
        glm::min(min, aabb.min),
            glm::max(max, aabb.max)
    };
}

template AABB<float> AABB<float>::expand(const AABB<float>& aabb)const;
template AABB<double> AABB<double>::expand(const AABB<double>& aabb)const;

template<typename Scalar>
CollisionDetection<Scalar>::CollisionDetection(const SolverData<Scalar>* solverData, const Context* context, const int _threadsPerBlock, size_t _maxNumQueries) :
    mpSolverData(solverData), mpContext(context), threadsPerBlock(_threadsPerBlock), maxNumQueries(_maxNumQueries), m_bvh(_threadsPerBlock)
{
    cudaMalloc(&dev_queries, maxNumQueries * sizeof(Query));

    cudaMalloc(&dev_numQueries, sizeof(size_t));
    cudaMemset(dev_numQueries, 0, sizeof(size_t));

    cudaMalloc(&dev_overflowFlag, sizeof(bool));
    mSqDisplay.create();
}

template<typename Scalar>
CollisionDetection<Scalar>::~CollisionDetection<Scalar>()
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

template<typename Scalar>
void CollisionDetection<Scalar>::PrepareRenderData()
{
    if (mpContext->guiData->QueryVis && numQueries > 0) {
        glm::vec3* pos;
        glm::vec4* col;
        MapDevicePosPtr(&pos, &col);
        thrust::device_ptr<const glm::tvec3<Scalar>> dvec3_ptr(mpSolverData->X);
        thrust::device_ptr<glm::vec3> vec3_ptr(pos);

        thrust::transform(dvec3_ptr, dvec3_ptr + numVerts, vec3_ptr,
            [] __host__ __device__(const glm::tvec3<Scalar> &d) {
            return glm::vec3(static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z));
        });
        cudaMemset(col, 0, numVerts * sizeof(glm::vec4));
        dim3 numBlocks((numQueries + threadsPerBlock - 1) / threadsPerBlock);
        processQueries << <numBlocks, threadsPerBlock >> > (dev_queries, numQueries, col);
        UnMapDevicePtr();
    }
    if (mpContext->guiData->BVHVis) {
        m_bvh.PrepareRenderData();
    }
}

template<typename Scalar>
void CollisionDetection<Scalar>::Draw(SurfaceShader* flatShaderProgram)
{
    if (mpContext->guiData->BVHVis)
        flatShaderProgram->draw(m_bvh, 0);
    if (mpContext->guiData->QueryVis)
        flatShaderProgram->drawPoints(*this);
    if (mpContext->guiData->QueryDebugMode) {
        glLineWidth(mpContext->guiData->LineWidth);
        flatShaderProgram->drawSingleQuery(GetSQDisplay(mpContext->guiData->CurrQueryId, mpSolverData->X,
            mpContext->guiData->QueryDirty ? mpContext->guiData->mPQuery : nullptr));
        mpContext->guiData->QueryDirty = false;
    }
}

template<typename Scalar>
SingleQueryDisplay& CollisionDetection<Scalar>::GetSQDisplay(int i, const glm::tvec3<Scalar>* X, Query* guiQuery)
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
        thrust::device_ptr<const glm::tvec3<Scalar>> dev_ptr_X(X);
        thrust::transform(dev_ptr_X + q.v0, dev_ptr_X + q.v0 + 1, dev_ptr, HighPtoFloatP<Scalar>());
        thrust::transform(dev_ptr_X + q.v1, dev_ptr_X + q.v1 + 1, dev_ptr + 1, HighPtoFloatP<Scalar>());
        thrust::transform(dev_ptr_X + q.v2, dev_ptr_X + q.v2 + 1, dev_ptr + 2, HighPtoFloatP<Scalar>());
        thrust::transform(dev_ptr_X + q.v3, dev_ptr_X + q.v3 + 1, dev_ptr + 3, HighPtoFloatP<Scalar>());

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
        thrust::device_ptr<const glm::tvec3<Scalar>> dev_ptr_X(X);
        thrust::transform(dev_ptr_X + q.v0, dev_ptr_X + q.v0 + 1, dev_vertPos, HighPtoFloatP<Scalar>());
        thrust::transform(dev_ptr_X + q.v1, dev_ptr_X + q.v1 + 1, dev_triPos, HighPtoFloatP<Scalar>());
        thrust::transform(dev_ptr_X + q.v2, dev_ptr_X + q.v2 + 1, dev_triPos + 1, HighPtoFloatP<Scalar>());
        thrust::transform(dev_ptr_X + q.v3, dev_ptr_X + q.v3 + 1, dev_triPos + 2, HighPtoFloatP<Scalar>());
        glm::vec3 v0Pos;
        cudaMemcpy(&v0Pos, vertPos, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        glm::vec3 normalPoint = v0Pos + q.normal * 10.f;
        cudaMemcpy(&pos[0], &v0Pos, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(&pos[1], &normalPoint, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        mSqDisplay.UnMapDevicePtr(&pos, &vertPos, &triPos);
    }
    return mSqDisplay;
}

template class CollisionDetection<float>;
template class CollisionDetection<double>;