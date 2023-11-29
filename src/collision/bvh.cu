#pragma once

#include <glm/glm.hpp>
#include <bvh.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <intersections.h>
#include <cuda_runtime.h>
#include <utilities.cuh>

__device__ AABB computeTetTrajBBox(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3,
    const glm::vec3& v4, const glm::vec3& v5, const glm::vec3& v6, const glm::vec3& v7)
{
    glm::vec3 min, max;
    min.x = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    min.y = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    min.z = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);
    max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);

    return AABB{ min, max };
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

bool isCollision(const glm::vec3& v, const AABB& box, float threshold = EPSILON) {
    glm::vec3 nearestPoint;
    nearestPoint.x = std::max(box.min.x, std::min(v.x, box.max.x));
    nearestPoint.y = std::max(box.min.y, std::min(v.y, box.max.y));
    nearestPoint.z = std::max(box.min.z, std::min(v.z, box.max.z));
    glm::vec3 diff = v - nearestPoint;
    float distanceSquared = glm::dot(diff, diff);
    return distanceSquared <= threshold;
}

__device__ float traverseTreePerVert(const BVHNode* nodes, const glm::vec3* Xs, const glm::vec3* XTilts, glm::vec3 X0, glm::vec3 XTilt, int& hitTetId)
{
    // record the closest intersection
    float closest = FLT_MAX;

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

        bool hitLeft = edgeBboxIntersectionTest(X0, XTilt, leftChild.bbox);
        bool hitRight = edgeBboxIntersectionTest(X0, XTilt, rightChild.bbox);
        if (hitLeft)
        {
            // check triangle intersection
            if (leftChild.isLeaf == 1)
            {
                float distance = tetrahedronTrajIntersectionTest(X0, XTilt, Xs, XTilts, leftChild.TetrahedronIndex);
                if (distance < closest)
                {
                    hitTetId = leftChild.TetrahedronIndex;
                    closest = distance;
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
                float distance = tetrahedronTrajIntersectionTest(X0, XTilt, Xs, XTilts, rightChild.TetrahedronIndex);
                if (distance < closest)
                {
                    hitTetId = rightChild.TetrahedronIndex;
                    closest = distance;
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

__device__ float traverseTreePerEdge(const BVHNode* nodes, const glm::vec3* Xs, const glm::vec3* XTilts, glm::vec3 X0, glm::vec3 XTilt, int& hitTetId)
{
    // record the closest intersection
    float closest = FLT_MAX;

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

        bool hitLeft = edgeBboxIntersectionTest(X0, XTilt, leftChild.bbox);
        bool hitRight = edgeBboxIntersectionTest(X0, XTilt, rightChild.bbox);
        if (hitLeft)
        {
            // check triangle intersection
            if (leftChild.isLeaf == 1)
            {
                float distance = tetrahedronTrajIntersectionTest(X0, XTilt, Xs, XTilts, leftChild.TetrahedronIndex);
                if (distance < closest)
                {
                    hitTetId = leftChild.TetrahedronIndex;
                    closest = distance;
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
                float distance = tetrahedronTrajIntersectionTest(X0, XTilt, Xs, XTilts, rightChild.TetrahedronIndex);
                if (distance < closest)
                {
                    hitTetId = rightChild.TetrahedronIndex;
                    closest = distance;
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


__global__ void detectCollisionCandidatesVertTriangleKern(int numVerts, const BVHNode* nodes, const GLuint* tetIds, const glm::vec3* Xs, const glm::vec3* XTilts, float* tI)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numVerts)
    {
        int hitTetId = -1;
        const glm::vec3& X = Xs[index];
        const glm::vec3& XTilt = XTilts[index];
        int tetId = tetIds[index];
        float distance = traverseTreePerVert(nodes, Xs, XTilts, X, XTilt, hitTetId);
        if (distance != -1)
        {
            tI[index] = distance;
        }
        else {
            tI[index] = 1;
        }
    }
}

__global__ void detectCollisionCandidatesEdgeEdgeKern(int numVerts, const BVHNode* nodes, const GLuint* tetIds, const glm::vec3* Xs, const glm::vec3* XTilts, float* tI)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numVerts)
    {
        int hitTetId = -1;
        const glm::vec3& X = Xs[index];
        const glm::vec3& XTilt = XTilts[index];
        int tetId = tetIds[index];
        float distance = traverseTreePerEdge(nodes, Xs, XTilts, X, XTilt, hitTetId);
        if (distance != -1)
        {
            tI[index] = distance;
        }
        else {
            tI[index] = 1;
        }
    }
}

float* BVH::DetectCollisionCandidates(const GLuint* Tet, const glm::vec3* Xs, const glm::vec3* XTilts, const GLuint* TetId) const
{
    int blockSize1d = 128;
    dim3 numblocks = (numVerts + blockSize1d - 1) / blockSize1d;
    detectCollisionCandidatesVertTriangleKern << <numblocks, blockSize1d >> > (numVerts, dev_BVHNodes, TetId, Xs, XTilts, dev_tIVertTri);
    detectCollisionCandidatesEdgeEdgeKern << <numblocks, blockSize1d >> > (numVerts, dev_BVHNodes, TetId, Xs, XTilts, dev_tIEdgeEdge);
    thrust::device_ptr<float> dev_tIVertTriPtr(dev_tIVertTri);
    thrust::device_ptr<float> dev_tIEdgeEdgePtr(dev_tIEdgeEdge);
    thrust::device_ptr<float> dev_tIPtr(dev_tI);
    thrust::transform(dev_tIVertTriPtr, dev_tIVertTriPtr + numVerts, dev_tIEdgeEdgePtr, dev_tIPtr, thrust::minimum<float>());
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

BVH::BVH(const int _threadsPerBlock, BuildMethodType _buildMethod) : threadsPerBlock(_threadsPerBlock), buildMethod(_buildMethod) {
}

BVH::~BVH()
{
    cudaFree(dev_BVHNodes);
    cudaFree(dev_tI);
    cudaFree(dev_tIVertTri);
    cudaFree(dev_tIEdgeEdge);

    cudaFree(dev_ready);
    cudaFree(dev_mortonCodes);
}

void BVH::Init(int _numTets, int _numVerts)
{
    numTets = _numTets;
    numVerts = _numVerts;
    numNodes = numTets * 2 - 1;
    cudaMalloc(&dev_BVHNodes, numNodes * sizeof(BVHNode));
    cudaMalloc((void**)&dev_tI, numVerts * sizeof(float));
    cudaMemset(dev_tI, 0, numVerts * sizeof(float));
    cudaMalloc((void**)&dev_tIVertTri, numVerts * sizeof(float));
    cudaMemset(dev_tIVertTri, 0, numVerts * sizeof(float));
    cudaMalloc((void**)&dev_tIEdgeEdge, numVerts * sizeof(float));
    cudaMemset(dev_tIEdgeEdge, 0, numVerts * sizeof(float));
    cudaMalloc(&dev_mortonCodes, numTets * sizeof(unsigned int));
    cudaMalloc(&dev_ready, numNodes * sizeof(int));
    createBVH(numNodes);
}