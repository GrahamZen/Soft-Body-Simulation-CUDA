#pragma once

#include <glm/glm.hpp>
#include <bvh.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <intersections.h>
#include <cuda_runtime.h>
#include <utilities.cuh>


struct QueryComparator {
    __host__ __device__
        bool operator()(const Query& lhs, const Query& rhs) const {
        if (lhs.type != rhs.type) return lhs.type < rhs.type;
        if (lhs.v0 != rhs.v0) return lhs.v0 < rhs.v0;
        if (lhs.v1 != rhs.v1) return lhs.v1 < rhs.v1;
        if (lhs.v2 != rhs.v2) return lhs.v2 < rhs.v2;
        return lhs.v3 < rhs.v3;
    }
};

struct QueryEquality {
    __host__ __device__
        bool operator()(const Query& lhs, const Query& rhs) const {
        return lhs.type == rhs.type &&
            lhs.v0 == rhs.v0 &&
            lhs.v1 == rhs.v1 &&
            lhs.v2 == rhs.v2 &&
            lhs.v3 == rhs.v3;
    }
};

struct IsUnknown {
    __host__ __device__
        bool operator()(const Query& query) const {
        return query.type == QueryType::UNKNOWN;
    }
};

__constant__ int edgeIndicesTable[12] = {
    0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3
};

template<typename T>
__host__ __device__ void sortThree(T& a, T& b, T& c) {
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
}

template<typename T>
__host__ __device__ void sortFour(T& a, T& b, T& c, T& d) {
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (c > d) swap(c, d);
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
}

__device__ void fillQuery(Query* query, int tetId, int tet2Id, const GLuint* tets) {
    for (int i = 0; i < 4; i++) {
        int vi = tets[tetId * 4 + i];
        int v20 = tets[tet2Id * 4 + 0];
        int v21 = tets[tet2Id * 4 + 1];
        int v22 = tets[tet2Id * 4 + 2];
        int v23 = tets[tet2Id * 4 + 3];
        query[i * 4].type = QueryType::VF;
        query[i * 4].v0 = vi;
        query[i * 4].v1 = v20;
        query[i * 4].v2 = v21;
        query[i * 4].v3 = v22;
        query[i * 4 + 1].type = QueryType::VF;
        query[i * 4 + 1].v0 = vi;
        query[i * 4 + 1].v1 = v20;
        query[i * 4 + 1].v2 = v21;
        query[i * 4 + 1].v3 = v23;
        query[i * 4 + 2].type = QueryType::VF;
        query[i * 4 + 2].v0 = vi;
        query[i * 4 + 2].v1 = v20;
        query[i * 4 + 2].v2 = v21;
        query[i * 4 + 2].v3 = v23;
        query[i * 4 + 3].type = QueryType::VF;
        query[i * 4 + 3].v0 = vi;
        query[i * 4 + 3].v1 = v21;
        query[i * 4 + 3].v2 = v22;
        query[i * 4 + 3].v3 = v23;
    }
    for (int i = 0; i < 6; i++) {
        int v0 = tets[tetId * 4 + edgeIndicesTable[i * 2 + 0]];
        int v1 = tets[tetId * 4 + edgeIndicesTable[i * 2 + 1]];
        int v20 = tets[tet2Id * 4 + 0];
        int v21 = tets[tet2Id * 4 + 1];
        int v22 = tets[tet2Id * 4 + 2];
        int v23 = tets[tet2Id * 4 + 3];
        query[i * 6 + 16].type = QueryType::EE;
        query[i * 6 + 16].v0 = v0;
        query[i * 6 + 16].v1 = v1;
        query[i * 6 + 16].v2 = v20;
        query[i * 6 + 16].v3 = v21;
        query[i * 6 + 17].type = QueryType::EE;
        query[i * 6 + 17].v0 = v0;
        query[i * 6 + 17].v1 = v1;
        query[i * 6 + 17].v2 = v20;
        query[i * 6 + 17].v3 = v22;
        query[i * 6 + 18].type = QueryType::EE;
        query[i * 6 + 18].v0 = v0;
        query[i * 6 + 18].v1 = v1;
        query[i * 6 + 18].v2 = v20;
        query[i * 6 + 18].v3 = v23;
        query[i * 6 + 19].type = QueryType::EE;
        query[i * 6 + 19].v0 = v0;
        query[i * 6 + 19].v1 = v1;
        query[i * 6 + 19].v2 = v21;
        query[i * 6 + 19].v3 = v22;
        query[i * 6 + 20].type = QueryType::EE;
        query[i * 6 + 20].v0 = v0;
        query[i * 6 + 20].v1 = v1;
        query[i * 6 + 20].v2 = v21;
        query[i * 6 + 20].v3 = v23;
        query[i * 6 + 21].type = QueryType::EE;
        query[i * 6 + 21].v0 = v0;
        query[i * 6 + 21].v1 = v1;
        query[i * 6 + 21].v2 = v22;
        query[i * 6 + 21].v3 = v23;
    }
}


__global__ void traverseTree(int numTets, const BVHNode* nodes, const GLuint* tets, Query* queries, int* queryCount, int maxQueries, bool* overflowFlag)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numTets) return;
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
                    Query* qBegin = &queries[qIdx];
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
                    Query* qBegin = &queries[qIdx];
                    if (qIdx < maxQueries) {
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

__global__ void sortEachQuery(int numQueries, Query* query)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        Query& q = query[index];
        if (q.type == QueryType::VF) {
            if (q.v0 == q.v1 || q.v0 == q.v2 || q.v0 == q.v3) {
                q.type = QueryType::UNKNOWN;
                return;
            }
            sortThree(q.v1, q.v2, q.v3);
        }
        else if (q.type == QueryType::EE) {
            sortTwo(q.v0, q.v1);
            sortTwo(q.v2, q.v3);
            if (q.v0 == q.v2 && q.v1 == q.v3) {
                q.type = QueryType::UNKNOWN;
            }
        }
        else {
            assert(false);
        }
    }
}

void removeDuplicates(Query* deviceQueries, int& deviceQueryCount) {
    thrust::device_ptr<Query> dev_ptr(deviceQueries);
    int numQueries = deviceQueryCount;

    auto new_end_remove = thrust::remove_if(dev_ptr, dev_ptr + numQueries, IsUnknown());
    numQueries = new_end_remove - dev_ptr;

    thrust::sort(dev_ptr, dev_ptr + numQueries, QueryComparator());

    auto new_end_unique = thrust::unique(dev_ptr, dev_ptr + numQueries, QueryEquality());

    deviceQueryCount = new_end_unique - dev_ptr;
}

int CollisionDetection::DetectCollisionCandidates(int numTets, const BVHNode* dev_BVHNodes, const GLuint* tets) {
    bool overflowHappened = false;
    bool overflow;
    dim3 numblocksTets = (numTets + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(deviceQueryCount, 0, sizeof(int));
    do {
        cudaMemset(deviceOverflowFlag, 0, sizeof(bool));
        traverseTree << <numblocksTets, threadsPerBlock >> > (numTets, dev_BVHNodes, tets, deviceQueries, deviceQueryCount, maxQueries, deviceOverflowFlag);
        cudaMemcpy(&overflow, deviceOverflowFlag, sizeof(bool), cudaMemcpyDeviceToHost);
        if (overflow) {
            std::cerr << "overflow" << std::endl;
            overflowHappened = true;
            maxQueries *= 2;
            cudaFree(deviceQueries);
            cudaMalloc(&deviceQueries, maxQueries * sizeof(Query));
            cudaMemset(deviceQueryCount, 0, sizeof(int));
        }
    } while (overflow);

    cudaMemcpy(&actualQueryCount, deviceQueryCount, sizeof(int), cudaMemcpyDeviceToHost);
    return actualQueryCount;
}

void CollisionDetection::BroadPhase(int numTets, const BVHNode* dev_BVHNodes, const GLuint* tets)
{
    DetectCollisionCandidates(numTets, dev_BVHNodes, tets);
    dim3 numBlocksQuery = (actualQueryCount + threadsPerBlock - 1) / threadsPerBlock;
    sortEachQuery << <numBlocksQuery, threadsPerBlock >> > (actualQueryCount, deviceQueries);
    removeDuplicates(deviceQueries, actualQueryCount);
}
