#pragma once

#include <glm/glm.hpp>
#include <bvh.h>
#include <intersections.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <utilities.cuh>
#include <simulationContext.h>

struct CompareQuery {
    __host__ __device__
        bool operator()(const Query& q1, const Query& q2) const {
        if (q1.type != q2.type)
        {
            return q1.type < q2.type;
        }
        else
        {
            if (q1.type == QueryType::VF)
            {
                if (q1.v0 == q2.v0) {
                    return q1.toi < q2.toi;
                }
                return q1.v0 < q2.v0;
            }
            else
            {
                if (q1.v0 == q2.v0 && q1.v1 == q2.v1) {
                    return q1.toi < q2.toi;
                }
                else
                {
                    if (q1.v0 == q2.v0)
                        return q1.v1 < q2.v1;
                    else
                        return q1.v0 < q2.v0;
                }
            }
        }
        
    }
};

struct EqualQuery {
    __host__ __device__
        bool operator()(const Query& q1, const Query& q2) const {
        if (q1.type == q2.type)
        {
            if (q1.type == QueryType::VF)
            {
                return q1.v0 == q2.v0;
            }
            else // if is EE
            {
                return (q1.v0 == q2.v0 && q1.v1 == q2.v1);
            }
        }
        return false;
    }
};

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
        q.toi = ccdCollisionTest(q, Xs, XTilts, normal);
        q.normal = normal;
    }
}

__global__ void storeTi(int numQueries, Query* queries, dataType* tI, glm::vec3* nors)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        Query& q = queries[index];
        
        if (q.type == QueryType::EE)
        {
            if (q.toi < 1.0f)
            {
                /*
                tI[q.v0] = q.toi;
                tI[q.v1] = q.toi;
                tI[q.v2] = q.toi;
                tI[q.v3] = q.toi;*/
                tI[q.v0] = 0.5f;
                tI[q.v1] = 0.5f;
                //tI[q.v2] = 0.5f;
                //tI[q.v3] = 0.5f;
                nors[q.v1] = q.normal;
                nors[q.v0] = q.normal;
            }
        }
        if (q.type == QueryType::VF)
        {
            if (q.toi < 1.0f)
            {
                tI[q.v0] = 0.5f;
                tI[q.v1] = 0.5f;
                tI[q.v2] = 0.5f;
                tI[q.v3] = 0.5f;
                nors[q.v0] = q.normal;
            }
        }
        /*
        if (q.type == QueryType::VF)
        {
            tI[q.v0] = q.toi;
            nors[q.v0] = q.normal;
        }*/
    }
}

__global__ void computeNewVel(int numQueries, const glm::vec3* Xs, const glm::vec3* XTilts, Query* queries, glm::vec3* Vs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        Query& q = queries[index];

        if (q.type == QueryType::EE)
        {
            if (q.toi < 1.0f)
            {
                float distance1 = glm::length(XTilts[q.v0] - Xs[q.v0]) * (1.0f - q.toi) + 0.001f;
                float distance2 = glm::length(XTilts[q.v1] - Xs[q.v1]) * (1.0f - q.toi) + 0.001f;
                glm::vec3 vel1 = 1.0f / distance1 * q.normal;
                glm::vec3 vel2 = 1.0f / distance2 * q.normal;

                atomicAdd(&Vs[q.v0][0], vel1[0]);
                atomicAdd(&Vs[q.v0][1], vel1[1]);
                atomicAdd(&Vs[q.v0][2], vel1[2]);

                atomicAdd(&Vs[q.v1][0], vel2[0]);
                atomicAdd(&Vs[q.v1][1], vel2[1]);
                atomicAdd(&Vs[q.v1][2], vel2[2]);
            }
        }
        if (q.type == QueryType::VF)
        {
            if (q.toi < 1.0f)
            {
                float distance = glm::length(XTilts[q.v0] - Xs[q.v0]) * (1.0f - q.toi) + 0.001f;
                glm::vec3 vel = 1.0f / distance * q.normal;

                atomicAdd(&Vs[q.v0][0], vel[0]);
                atomicAdd(&Vs[q.v0][1], vel[1]);
                atomicAdd(&Vs[q.v0][2], vel[2]);

                atomicAdd(&Vs[q.v1][0], -vel[0]);
                atomicAdd(&Vs[q.v1][1], -vel[1]);
                atomicAdd(&Vs[q.v1][2], -vel[2]);

                atomicAdd(&Vs[q.v2][0], -vel[0]);
                atomicAdd(&Vs[q.v2][1], -vel[1]);
                atomicAdd(&Vs[q.v2][2], -vel[2]);

                atomicAdd(&Vs[q.v3][0], -vel[0]);
                atomicAdd(&Vs[q.v3][1], -vel[1]);
                atomicAdd(&Vs[q.v3][2], -vel[2]);
            }
        }
    }
}

void CollisionDetection::NarrowPhase(const glm::vec3* Xs, const glm::vec3* XTilts, dataType*& tI, glm::vec3*& nors)
{
    dim3 numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    detectCollisionNarrow << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries, Xs, XTilts);
    thrust::device_ptr<Query> dev_queriesPtr(dev_queries);

    thrust::sort(dev_queriesPtr, dev_queriesPtr + numQueries, CompareQuery());
    auto new_end = thrust::unique(dev_queriesPtr, dev_queriesPtr + numQueries, EqualQuery());
    int numQueries = new_end - dev_queriesPtr;
    numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    //computeNewVel << <numBlocksQuery, threadsPerBlock >> > (numQueries, Xs, XTilts, dev_queries, mPSimContext->dev_Vs);
    storeTi << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries, tI, nors);
    cudaDeviceSynchronize();
}