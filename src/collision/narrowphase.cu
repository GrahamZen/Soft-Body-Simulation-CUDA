#pragma once

#include <glm/glm.hpp>
#include <bvh.cuh>
#include <intersections.h>
#include <cuda_runtime.h>
#include <utilities.cuh>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

struct CompareQuery {
    __host__ __device__
        bool operator()(const Query& q1, const Query& q2) const {
        if (q1.v0 == q2.v0) {
            return q1.toi < q2.toi;
        }
        return q1.v0 < q2.v0;
    }
};

struct EqualQuery {
    __host__ __device__
        bool operator()(const Query& q1, const Query& q2) const {
        return q1.v0 == q2.v0;
    }
};

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

__global__ void detectCollisionNarrow(int numQueries, Query* queries, const glm::vec3* Xs, const glm::vec3* XTilts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        glmVec3 normal;
        Query& q = queries[index];
        if (q.type == QueryType::VF) {
            GLuint targetVIdx = q.v0;
            q.toi = ccdCollisionTest(q, Xs, XTilts, normal);
            q.normal = normal;
        }
    }
}

__global__ void storeTi(int numQueries, Query* queries, dataType* tI, glm::vec3* nors)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        Query& q = queries[index];
        tI[q.v0] = q.toi;
        nors[q.v0] = q.normal;
    }
}

void CollisionDetection::NarrowPhase(const glm::vec3* Xs, const glm::vec3* XTilts, dataType* tI, glm::vec3* nors)
{
    dim3 numBlocksQuery = (actualQueryCount + threadsPerBlock - 1) / threadsPerBlock;
    detectCollisionNarrow << <numBlocksQuery, threadsPerBlock >> > (actualQueryCount, deviceQueries, Xs, XTilts);
    thrust::device_ptr<Query> dev_queries(deviceQueries);

    thrust::sort(dev_queries, dev_queries + actualQueryCount, CompareQuery());

    auto new_end = thrust::unique(dev_queries, dev_queries + actualQueryCount, EqualQuery());

    int actualQueryCount = new_end - dev_queries;
    numBlocksQuery = (actualQueryCount + threadsPerBlock - 1) / threadsPerBlock;
    storeTi << <numBlocksQuery, threadsPerBlock >> > (actualQueryCount, deviceQueries, tI, nors);
}