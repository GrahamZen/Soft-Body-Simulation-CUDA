#pragma once

#include <utilities.cuh>
#include <collision/bvh.cuh>
#include <collision/bvh.h>
#include <simulation/simulationContext.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <collision/intersections.h>
#include <cuda_runtime.h>

// build the bounding box and morton code for each SoftBody
template<typename HighP>
__global__ void buildLeafMorton(int startIndex, int numTri, dataType minX, dataType minY, dataType minZ,
    dataType maxX, dataType maxY, dataType maxZ, const indexType* tet, const glm::tvec3<HighP>* X, const glm::tvec3<HighP>* XTilde, BVHNode* leafNodes,
    unsigned int* mortonCodes)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < numTri)
    {
        int leafPos = ind + numTri - 1;
        leafNodes[leafPos].bbox = computeTetTrajBBox(X[tet[ind * 4]], X[tet[ind * 4 + 1]], X[tet[ind * 4 + 2]], X[tet[ind * 4 + 3]],
            XTilde[tet[ind * 4]], XTilde[tet[ind * 4 + 1]], XTilde[tet[ind * 4 + 2]], XTilde[tet[ind * 4 + 3]]);
        leafNodes[leafPos].isLeaf = 1;
        leafNodes[leafPos].leftIndex = -1;
        leafNodes[leafPos].rightIndex = -1;
        leafNodes[leafPos].TetrahedronIndex = ind;
        mortonCodes[ind + startIndex] = genMortonCode(leafNodes[ind + numTri - 1].bbox, glmVec3(minX, minY, minZ), glmVec3(maxX, maxY, maxZ));
    }
}

template<typename HighP>
void BVH::BuildBVHTree(BuildType buildType, const AABB& ctxAABB, int numTets, const glm::tvec3<HighP>* X, glm::tvec3<HighP>* XTilde, const indexType* tets)
{
    cudaMemset(dev_BVHNodes, 0, (numTets * 2 - 1) * sizeof(BVHNode));

    buildLeafMorton << <numblocksTets, threadsPerBlock >> > (0, numTets, ctxAABB.min.x, ctxAABB.min.y, ctxAABB.min.z, ctxAABB.max.x, ctxAABB.max.y, ctxAABB.max.z,
        tets, X, XTilde, dev_BVHNodes, dev_mortonCodes);

    thrust::stable_sort_by_key(thrust::device, dev_mortonCodes, dev_mortonCodes + numTets, dev_BVHNodes + numTets - 1);

    buildSplitList << <numblocksTets, threadsPerBlock >> > (numTets, dev_mortonCodes, dev_BVHNodes);

    BuildBBoxes(buildType);

    cudaMemset(dev_mortonCodes, 0, numTets * sizeof(unsigned int));
    cudaMemset(dev_ready, 0, (numTets - 1) * sizeof(ReadyFlagType));
    cudaMemset(&dev_ready[numTets - 1], 1, numTets * sizeof(ReadyFlagType));
}

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

__device__ void fillQuery(Query* query, int tetId, int tet2Id, const indexType* tets) {
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
        query[i * 4 + 2].v2 = v22;
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


__global__ void traverseTree(int numTets, const BVHNode* nodes, const indexType* tets, const indexType* tetFathers, Query* queries, size_t* queryCount, size_t maxNumQueries, bool* overflowFlag)
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
                if (tetFathers[myNode.TetrahedronIndex] != tetFathers[leftChild.TetrahedronIndex] && myNode.TetrahedronIndex != leftChild.TetrahedronIndex) {
                    int qIdx = atomicAdd(queryCount, 36 + 16);
                    if (qIdx + 52 < maxNumQueries) {
                        Query* qBegin = &queries[qIdx];
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
                if (tetFathers[myNode.TetrahedronIndex] != tetFathers[rightChild.TetrahedronIndex] && myNode.TetrahedronIndex != rightChild.TetrahedronIndex) {
                    int qIdx = atomicAdd(queryCount, 36 + 16);
                    if (qIdx + 52 < maxNumQueries) {
                        Query* qBegin = &queries[qIdx];
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

bool CollisionDetection::DetectCollisionCandidates(const BVHNode* dev_BVHNodes) {
    bool overflowHappened = false;
    bool overflow;
    dim3 numblocksTets = (mPSimContext->mSolverData.numTets + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(dev_numQueries, 0, sizeof(size_t));
    do {
        overflow = false;
        cudaMemset(dev_overflowFlag, 0, sizeof(bool));
        traverseTree << <numblocksTets, threadsPerBlock >> > (mPSimContext->mSolverData.numTets, dev_BVHNodes, mPSimContext->mSolverData.Tet, mPSimContext->dev_TetFathers, dev_queries, dev_numQueries, maxNumQueries, dev_overflowFlag);
        cudaMemcpy(&overflow, dev_overflowFlag, sizeof(bool), cudaMemcpyDeviceToHost);
        if (overflow) {
            overflowHappened = true;
            maxNumQueries *= 2;
            std::cerr << "Query buffer overflow, resizing to " << maxNumQueries << std::endl;
            if (maxNumQueries > 1 << 31) {
                std::cerr << "Number of queries exceeds 2^31, aborting" << std::endl;
                exit(1);
                return false;
            }
            cudaFree(dev_queries);
            cudaMalloc(&dev_queries, maxNumQueries * sizeof(Query));
            cudaMemset(dev_numQueries, 0, sizeof(size_t));
        }
    } while (overflow);

    cudaMemcpy(&numQueries, dev_numQueries, sizeof(size_t), cudaMemcpyDeviceToHost);
    return numQueries > 0;
}

void CollisionDetection::Init(int numTets, int numVerts, int maxThreads)
{
    createQueries(numVerts);
    m_bvh.Init(numTets, numVerts, maxThreads);
}

__global__ void sortEachQuery(size_t numQueries, Query* query)
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

void removeDuplicates(Query* dev_queries, size_t& dev_numQueries) {
    thrust::device_ptr<Query> dev_ptr(dev_queries);
    size_t numQueries = dev_numQueries;

    auto new_end_remove = thrust::remove_if(dev_ptr, dev_ptr + numQueries, IsUnknown());
    numQueries = new_end_remove - dev_ptr;

    thrust::sort(dev_ptr, dev_ptr + numQueries, QueryComparator());

    auto new_end_unique = thrust::unique(dev_ptr, dev_ptr + numQueries, QueryEquality());

    dev_numQueries = new_end_unique - dev_ptr;
}

template<typename HighP>
struct MinOp {
    __host__ __device__
        glm::tvec3<HighP> operator()(const glm::tvec3<HighP>& a, const glm::tvec3<HighP>& b) const {
        return glm::min(a, b);
    }
};

template<typename HighP>
struct MaxOp {
    __host__ __device__
        glm::tvec3<HighP> operator()(const glm::tvec3<HighP>& a, const glm::tvec3<HighP>& b) const {
        return glm::max(a, b);
    }
};

template<typename HighP>
AABB computeBoundingBox(const thrust::device_ptr<glm::tvec3<HighP>>& begin, const thrust::device_ptr<glm::tvec3<HighP>>& end) {
    glm::tvec3<HighP> min = thrust::reduce(begin, end, glm::tvec3<HighP>(FLT_MAX), MinOp<HighP>());
    glm::tvec3<HighP> max = thrust::reduce(begin, end, glm::tvec3<HighP>(-FLT_MAX), MaxOp<HighP>());

    return AABB{ min, max };
}

AABB CollisionDetection::GetAABB() const
{
    using XType = typename std::decay<decltype(*mPSimContext->mSolverData.X)>::type;
    thrust::device_ptr<XType> dev_ptr(mPSimContext->mSolverData.X);
    thrust::device_ptr<XType> dev_ptrTildes(mPSimContext->mSolverData.XTilde);
    return computeBoundingBox(dev_ptr, dev_ptr + numVerts).expand(computeBoundingBox(dev_ptrTildes, dev_ptrTildes + numVerts));
}

bool CollisionDetection::BroadPhase()
{
    const std::string buildTypeStr = buildType == BVH::BuildType::Cooperative ? "Cooperative" : buildType == BVH::BuildType::Atomic ? "Atomic" : "Serial";
    m_bvh.BuildBVHTree(buildType, GetAABB(), mPSimContext->mSolverData.numTets, mPSimContext->mSolverData.X, mPSimContext->mSolverData.XTilde, mPSimContext->mSolverData.Tet);

    if (!DetectCollisionCandidates(m_bvh.GetBVHNodes())) {
        count = 0;
        return false;
    }
    dim3 numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    sortEachQuery << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries);
    removeDuplicates(dev_queries, numQueries);
    count = numVerts;
    return true;
}
