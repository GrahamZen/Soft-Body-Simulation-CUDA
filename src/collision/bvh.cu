#pragma once

#include <glm/glm.hpp>
#include <bvh.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <intersections.h>
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

bool isCollision(const glm::vec3& v, const AABB& box, dataType threshold = EPSILON) {
    glm::vec3 nearestPoint;
    nearestPoint.x = std::max(box.min.x, std::min(v.x, box.max.x));
    nearestPoint.y = std::max(box.min.y, std::min(v.y, box.max.y));
    nearestPoint.z = std::max(box.min.z, std::min(v.z, box.max.z));
    glmVec3 diff = v - nearestPoint;
    dataType distanceSquared = glm::dot(diff, diff);
    return distanceSquared <= threshold;
}

__device__ dataType traverseTree(int vertIdx, const BVHNode* nodes, const GLuint* tets, const glm::vec3* Xs, const glm::vec3* XTilts, int& hitTetId, glm::vec3& nor)
{
    // record the closest intersection
    dataType closest = 1;
    const glmVec3 x0 = Xs[vertIdx];
    const glmVec3 xTilt = XTilts[vertIdx];
    int bvhStart = 0;
    int stack[64];
    int stackPtr = 0;
    int bvhPtr = bvhStart;
    stack[stackPtr++] = bvhStart;

    while (stackPtr)
    {
        bvhPtr = stack[--stackPtr];
        BVHNode currentNode = nodes[bvhPtr];
        // all the left and right indexes are 0
        BVHNode leftChild = nodes[currentNode.leftIndex + bvhStart];
        BVHNode rightChild = nodes[currentNode.rightIndex + bvhStart];

        bool hitLeft = edgeBboxIntersectionTest(x0, xTilt, leftChild.bbox);
        bool hitRight = edgeBboxIntersectionTest(x0, xTilt, rightChild.bbox);
        if (hitLeft)
        {
            // check triangle intersection
            if (leftChild.isLeaf == 1)
            {
                if (tets[leftChild.TetrahedronIndex * 4 + 0] != vertIdx && tets[leftChild.TetrahedronIndex * 4 + 1] != vertIdx &&
                    tets[leftChild.TetrahedronIndex * 4 + 2] != vertIdx && tets[leftChild.TetrahedronIndex * 4 + 3] != vertIdx) {
                    dataType distance = tetrahedronTrajIntersectionTest(tets, x0, xTilt, Xs, XTilts, leftChild.TetrahedronIndex, nor);
                    if (distance < closest)
                    {
                        hitTetId = leftChild.TetrahedronIndex;
                        closest = distance;
                    }
                }
            }
            else
            {
                stack[stackPtr++] = currentNode.leftIndex + bvhStart;
            }

        }
        if (hitRight)
        {
            // check triangle intersection
            if (rightChild.isLeaf == 1)
            {
                if (tets[rightChild.TetrahedronIndex * 4 + 0] != vertIdx && tets[rightChild.TetrahedronIndex * 4 + 1] != vertIdx &&
                    tets[rightChild.TetrahedronIndex * 4 + 2] != vertIdx && tets[rightChild.TetrahedronIndex * 4 + 3] != vertIdx) {
                    dataType distance = tetrahedronTrajIntersectionTest(tets, x0, xTilt, Xs, XTilts, rightChild.TetrahedronIndex, nor);
                    if (distance < closest)
                    {
                        hitTetId = rightChild.TetrahedronIndex;
                        closest = distance;
                    }
                }
            }
            else
            {
                stack[stackPtr++] = currentNode.rightIndex + bvhStart;
            }

        }
    }
    return closest;
}

__constant__ int edgeIndicesTable[12] = {
    0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3
};
__device__ void fillQuery(Query* query, int tetId, int tet2Id, const GLuint* tets) {
    for (int i = 0; i < 4; i++) {
        query[i * 4].type = QueryType::VF;
        query[i * 4].v0 = tets[tetId * 4 + i];
        query[i * 4].v1 = tets[tet2Id * 4 + 0];
        query[i * 4].v2 = tets[tet2Id * 4 + 1];
        query[i * 4].v3 = tets[tet2Id * 4 + 2];
        query[i * 4 + 1].type = QueryType::VF;
        query[i * 4 + 1].v0 = tets[tetId * 4 + i];
        query[i * 4 + 1].v1 = tets[tet2Id * 4 + 0];
        query[i * 4 + 1].v2 = tets[tet2Id * 4 + 1];
        query[i * 4 + 1].v3 = tets[tet2Id * 4 + 3];
        query[i * 4 + 2].type = QueryType::VF;
        query[i * 4 + 2].v0 = tets[tetId * 4 + i];
        query[i * 4 + 2].v1 = tets[tet2Id * 4 + 0];
        query[i * 4 + 2].v2 = tets[tet2Id * 4 + 1];
        query[i * 4 + 2].v3 = tets[tet2Id * 4 + 3];
        query[i * 4 + 3].type = QueryType::VF;
        query[i * 4 + 3].v0 = tets[tetId * 4 + i];
        query[i * 4 + 3].v1 = tets[tet2Id * 4 + 1];
        query[i * 4 + 3].v2 = tets[tet2Id * 4 + 2];
        query[i * 4 + 3].v3 = tets[tet2Id * 4 + 3];
    }
    for (int i = 0; i < 6; i++) {
        int v0 = tets[tetId * 4 + edgeIndicesTable[i * 2 + 0]];
        int v1 = tets[tetId * 4 + edgeIndicesTable[i * 2 + 1]];
        query[i * 6 + 16].type = QueryType::EE;
        query[i * 6 + 16].v0 = v0;
        query[i * 6 + 16].v1 = v1;
        query[i * 6 + 16].v2 = tets[tet2Id * 4 + 0];
        query[i * 6 + 16].v3 = tets[tet2Id * 4 + 1];
        query[i * 6 + 17].type = QueryType::EE;
        query[i * 6 + 17].v0 = v0;
        query[i * 6 + 17].v1 = v1;
        query[i * 6 + 17].v2 = tets[tet2Id * 4 + 0];
        query[i * 6 + 17].v3 = tets[tet2Id * 4 + 2];
        query[i * 6 + 18].type = QueryType::EE;
        query[i * 6 + 18].v0 = v0;
        query[i * 6 + 18].v1 = v1;
        query[i * 6 + 18].v2 = tets[tet2Id * 4 + 0];
        query[i * 6 + 18].v3 = tets[tet2Id * 4 + 3];
        query[i * 6 + 19].type = QueryType::EE;
        query[i * 6 + 19].v0 = v0;
        query[i * 6 + 19].v1 = v1;
        query[i * 6 + 19].v2 = tets[tet2Id * 4 + 1];
        query[i * 6 + 19].v3 = tets[tet2Id * 4 + 2];
        query[i * 6 + 20].type = QueryType::EE;
        query[i * 6 + 20].v0 = v0;
        query[i * 6 + 20].v1 = v1;
        query[i * 6 + 20].v2 = tets[tet2Id * 4 + 1];
        query[i * 6 + 20].v3 = tets[tet2Id * 4 + 3];
        query[i * 6 + 21].type = QueryType::EE;
        query[i * 6 + 21].v0 = v0;
        query[i * 6 + 21].v1 = v1;
        query[i * 6 + 21].v2 = tets[tet2Id * 4 + 2];
        query[i * 6 + 21].v3 = tets[tet2Id * 4 + 3];;
    }
}


__global__ void traverseTree(int numTets, const BVHNode* nodes, const GLuint* tets, Query* queries, int* queryCount, int maxQueries, bool* overflowFlag)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int leafIdx = index + numTets - 1;
    const BVHNode myNode = nodes[leafIdx];
    // record the closest intersection
    dataType closest = 1;
    int bvhStart = 0;
    int stack[64];
    int stackPtr = 0;
    int bvhPtr = bvhStart;
    stack[stackPtr++] = bvhStart;

    while (stackPtr)
    {
        bvhPtr = stack[--stackPtr];
        BVHNode currentNode = nodes[bvhPtr];
        // all the left and right indexes are 0
        BVHNode leftChild = nodes[currentNode.leftIndex + bvhStart];
        BVHNode rightChild = nodes[currentNode.rightIndex + bvhStart];

        bool hitLeft = bboxIntersectionTest(myNode.bbox, leftChild.bbox);
        bool hitRight = bboxIntersectionTest(myNode.bbox, rightChild.bbox);
        if (hitLeft)
        {
            // check triangle intersection
            if (leftChild.isLeaf == 1)
            {
                // 4 faces * 4 verts + 6 edges * 6 edges
                if (myNode.TetrahedronIndex != leftChild.TetrahedronIndex) {
                    int qIdx = atomicAdd(queryCount, 36 + 16);
                    Query* qBegin = &queries[*queryCount - 52];
                    printf("--------------------------------\ntet%d hit tet%d\n", myNode.TetrahedronIndex, rightChild.TetrahedronIndex);
                    printf("mybbox = AABB{glmVec3(%f, %f, %f), glmVec3(%f, %f, %f)}; bbox = AABB{glmVec3(%f, %f, %f), glmVec3(%f, %f, %f)};\n--------------------------------\n",
                        myNode.bbox.min.x, myNode.bbox.min.y, myNode.bbox.min.z,
                        myNode.bbox.max.x, myNode.bbox.max.y, myNode.bbox.max.z,
                        rightChild.bbox.min.x, rightChild.bbox.min.y, rightChild.bbox.min.z,
                        rightChild.bbox.max.x, rightChild.bbox.max.y, rightChild.bbox.max.z);
                    if (qIdx < maxQueries) {
                        fillQuery(qBegin, myNode.TetrahedronIndex, leftChild.TetrahedronIndex, tets);
                    }
                    else {
                        *overflowFlag = true;
                        return;
                    }
                }
            }
            else
            {
                stack[stackPtr++] = currentNode.leftIndex + bvhStart;
            }

        }
        if (hitRight)
        {
            // check triangle intersection
            if (rightChild.isLeaf == 1)
            {
                if (myNode.TetrahedronIndex != rightChild.TetrahedronIndex) {
                    int qIdx = atomicAdd(queryCount, 36 + 16);
                    Query* qBegin = &queries[*queryCount - 52];
                    printf("--------------------------------\ntet%d hit tet%d\n", myNode.TetrahedronIndex, rightChild.TetrahedronIndex);
                    printf("mybbox = AABB{glmVec3(%f, %f, %f), glmVec3(%f, %f, %f)}; bbox = AABB{glmVec3(%f, %f, %f), glmVec3(%f, %f, %f)};\n--------------------------------\n",
                        myNode.bbox.min.x, myNode.bbox.min.y, myNode.bbox.min.z,
                        myNode.bbox.max.x, myNode.bbox.max.y, myNode.bbox.max.z,
                        rightChild.bbox.min.x, rightChild.bbox.min.y, rightChild.bbox.min.z,
                        rightChild.bbox.max.x, rightChild.bbox.max.y, rightChild.bbox.max.z);
                    if (qIdx < maxQueries && myNode.TetrahedronIndex != rightChild.TetrahedronIndex) {
                        fillQuery(qBegin, myNode.TetrahedronIndex, rightChild.TetrahedronIndex, tets);
                    }
                    else {
                        *overflowFlag = true;
                        return;
                    }
                }
            }
            else
            {
                stack[stackPtr++] = currentNode.rightIndex + bvhStart;
            }

        }
    }
}


__global__ void detectCollisionCandidatesKern(int numVerts, const BVHNode* nodes, const GLuint* tets, const glm::vec3* Xs, const glm::vec3* XTilts, dataType* tI, glm::vec3* nors)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numVerts)
    {
        int hitTetId = -1;
        tI[index] = traverseTree(index, nodes, tets, Xs, XTilts, hitTetId, nors[index]);
    }
}

dataType* BVH::DetectCollisionCandidates(const GLuint* edges, const GLuint* tets, const glm::vec3* Xs, const glm::vec3* XTilts, glm::vec3* nors)
{
    dim3 numblocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    dim3 numblocksTets = (numTets + threadsPerBlock - 1) / threadsPerBlock;
    bool overflowHappened = false;
    bool overflow;
    cudaMemset(deviceQueryCount, 0, sizeof(int)); // 重置计数器
    do {
        cudaMemset(deviceOverflowFlag, 0, sizeof(bool));
        traverseTree << <numblocksTets, threadsPerBlock >> > (numTets, dev_BVHNodes, tets, deviceQueries, deviceQueryCount, maxQueries, deviceOverflowFlag);
        // 检查是否溢出
        cudaMemcpy(&overflow, deviceOverflowFlag, sizeof(bool), cudaMemcpyDeviceToHost);
        if (overflow) {
            std::cerr << "overflow" << std::endl;
            overflowHappened = true;
            maxQueries *= 2; // 或选择其他增长策略
            cudaFree(deviceQueries);
            cudaMalloc(&deviceQueries, maxQueries * sizeof(Query));
            cudaMemset(deviceQueryCount, 0, sizeof(int)); // 重置计数器
        }
    } while (overflow);

    // 从 deviceQueryCount 中读取实际的查询数量
    int actualQueryCount;
    cudaMemcpy(&actualQueryCount, deviceQueryCount, sizeof(int), cudaMemcpyDeviceToHost);
    inspectQuerys(deviceQueries, actualQueryCount);

    // 如果发生溢出，你可能需要重新处理数据
    if (overflowHappened) {
        // 处理溢出情况...
    }


    detectCollisionCandidatesKern << <numblocks, threadsPerBlock >> > (numVerts, dev_BVHNodes, tets, Xs, XTilts, dev_tI, nors);
    return dev_tI;
}

void BVH::PrepareRenderData()
{
    glm::vec3* pos;
    Wireframe::mapDevicePosPtr(&pos);
    dim3 numThreadsPerBlock(numNodes / threadsPerBlock + 1);
    populateBVHNodeAABBPos << <numThreadsPerBlock, threadsPerBlock >> > (dev_BVHNodes, pos, numNodes);
    Wireframe::unMapDevicePtr();
}

BVH::BVH(const int _threadsPerBlock, int _maxQueries) : threadsPerBlock(_threadsPerBlock), maxQueries(_maxQueries) {
}

BVH::~BVH()
{
    cudaFree(dev_BVHNodes);
    cudaFree(dev_tI);
    cudaFree(dev_indicesToReport);

    cudaFree(dev_ready);
    cudaFree(dev_mortonCodes);

    cudaFree(deviceQueries);

    cudaFree(deviceQueryCount);

    cudaFree(deviceOverflowFlag);

}