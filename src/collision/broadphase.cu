#pragma once

#include <utilities.cuh>
#include <collision/bvh.cuh>
#include <collision/bvh.h>
#include <simulation/simulationContext.h>
#include <collision/intersections.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <cuda_runtime.h>

//input the aabb box of a Tetrahedron
//generate a 30-bit morton code
template<typename Scalar>
__device__ unsigned int genMortonCode(AABB<Scalar> bbox, glm::tvec3<Scalar> geoMin, glm::tvec3<Scalar> geoMax)
{
    Scalar x = (bbox.min.x + bbox.max.x) * 0.5f;
    Scalar y = (bbox.min.y + bbox.max.y) * 0.5f;
    Scalar z = (bbox.min.y + bbox.max.y) * 0.5f;
    Scalar normalizedX = (x - geoMin.x) / (geoMax.x - geoMin.x);
    Scalar normalizedY = (y - geoMin.y) / (geoMax.y - geoMin.y);
    Scalar normalizedZ = (z - geoMin.z) / (geoMax.z - geoMin.z);

    normalizedX = glm::min(glm::max(normalizedX * 1024.0, 0.0), 1023.0);
    normalizedY = glm::min(glm::max(normalizedY * 1024.0, 0.0), 1023.0);
    normalizedZ = glm::min(glm::max(normalizedZ * 1024.0, 0.0), 1023.0);

    unsigned int xx = expandBits((unsigned int)normalizedX);
    unsigned int yy = expandBits((unsigned int)normalizedY);
    unsigned int zz = expandBits((unsigned int)normalizedZ);

    return xx * 4 + yy * 2 + zz;
}

template unsigned int genMortonCode<float>(AABB<float> bbox, glm::tvec3<float> geoMin, glm::tvec3<float> geoMax);
template unsigned int genMortonCode<double>(AABB<double> bbox, glm::tvec3<double> geoMin, glm::tvec3<double> geoMax);

__device__ unsigned long long expandMorton(int index, unsigned int mortonCode)
{
    unsigned long long exMortonCode = mortonCode;
    exMortonCode <<= 32;
    exMortonCode += index;
    return exMortonCode;
}

/**
* please sort the morton code first then get split pairs
thrust::stable_sort_by_key(mortonCodes, mortonCodes + TetrahedronCount, TetrahedronIndex);*/

//total input is a 30 x N matrix
//currentIndex is between 0 - N-1
//the input morton codes should be in the reduced form, no same elements are expected to appear twice!
__device__ int getSplit(unsigned int* mortonCodes, unsigned int currIndex, int nextIndex, unsigned int bound)
{
    if (nextIndex < 0 || nextIndex >= bound)
        return -1;
    //NOTE: if use small size model, this step can be skipped
    // just to ensure the morton codes are unique!
    //unsigned int mask = mortonCodes[currIndex] ^ mortonCodes[nextIndex];
    unsigned long long mask = expandMorton(currIndex, mortonCodes[currIndex]) ^ expandMorton(nextIndex, mortonCodes[nextIndex]);
    // __clzll gives the number of consecutive zero bits in that number
    // this gives us the index of the most significant bit between the two numbers
    int commonPrefix = __clzll(mask);
    return commonPrefix;
}

//input the unique morton code
//codeCount is the size of the unique morton code
//splitList is 30 x N list
// the size of unique morton is less than 2^30 : [1, 2^30]
template<typename Scalar>
__global__ void buildSplitList(int codeCount, unsigned int* uniqueMorton, BVHNode<Scalar>* nodes)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < codeCount - 1)
    {
        int sign = getSign(getSplit(uniqueMorton, ind, ind + 1, codeCount) - getSplit(uniqueMorton, ind, ind - 1, codeCount));
        int dMin = getSplit(uniqueMorton, ind, ind - sign, codeCount);
        int lenMax = 2;
        int k = getSplit(uniqueMorton, ind, ind + lenMax * sign, codeCount);
        while (k > dMin)
        {
            lenMax *= 2;
            k = getSplit(uniqueMorton, ind, ind + lenMax * sign, codeCount);
        }

        int len = 0;
        int last = lenMax >> 1;
        while (last > 0)
        {
            int tmp = ind + (len + last) * sign;
            int diff = getSplit(uniqueMorton, ind, tmp, codeCount);
            if (diff > dMin)
            {
                len = len + last;
            }
            last >>= 1;
        }
        //last in range
        int j = ind + len * sign;

        int currRange = getSplit(uniqueMorton, ind, j, codeCount);
        int split = 0;
        do {
            len = (len + 1) >> 1;
            if (getSplit(uniqueMorton, ind, ind + (split + len) * sign, codeCount) > currRange)
            {
                split += len;
            }
        } while (len > 1);

        int tmp = ind + split * sign + glm::min(sign, 0);

        if (glm::min(ind, j) == tmp)
        {
            //leaf node
            // the number of internal nodes is N - 1
            nodes[ind].leftIndex = tmp + codeCount - 1;
            nodes[tmp + codeCount - 1].parent = ind;
        }
        else
        {
            // internal node
            nodes[ind].leftIndex = tmp;
            nodes[tmp].parent = ind;
        }
        if (glm::max(ind, j) == tmp + 1)
        {
            nodes[ind].rightIndex = tmp + codeCount;
            nodes[tmp + codeCount].parent = ind;
        }
        else
        {
            nodes[ind].rightIndex = tmp + 1;
            nodes[tmp + 1].parent = ind;
        }
    }

}

// build the bounding box and morton code for each SoftBody
template<typename Scalar>
__global__ void buildLeafMorton(int startIndex, int numTri, Scalar minX, Scalar minY, Scalar minZ,
    Scalar maxX, Scalar maxY, Scalar maxZ, const indexType* tet, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, BVHNode<Scalar>* leafNodes,
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
        mortonCodes[ind + startIndex] = genMortonCode(leafNodes[ind + numTri - 1].bbox, glm::tvec3<Scalar>(minX, minY, minZ), glm::tvec3<Scalar>(maxX, maxY, maxZ));
    }
}

template<typename Scalar>
void BVH<Scalar>::BuildBVHTree(BuildType buildType, const AABB<Scalar>& ctxAABB, int numTets, const glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, const indexType* tets)
{
    cudaMemset(dev_BVHNodes, 0, (numTets * 2 - 1) * sizeof(BVHNode<Scalar>));

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


template<typename Scalar>
__global__ void traverseTree(int numTets, const BVHNode<Scalar>* nodes, const indexType* tets, const indexType* tetFathers, Query* queries, size_t* queryCount, size_t maxNumQueries, bool* overflowFlag)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numTets) return;
    int leafIdx = index + numTets - 1;
    const BVHNode<Scalar> myNode = nodes[leafIdx];
    // record the closest intersection
    Scalar closest = 1;
    int bvhStart = 0;
    int stack[64];
    int stackPtr = 0;
    int bvhPtr = bvhStart;
    stack[stackPtr++] = bvhStart;

    while (stackPtr)
    {
        bvhPtr = stack[--stackPtr];
        BVHNode<Scalar> currentNode = nodes[bvhPtr];
        // all the left and right indexes are 0
        BVHNode<Scalar> leftChild = nodes[currentNode.leftIndex + bvhStart];
        BVHNode<Scalar> rightChild = nodes[currentNode.rightIndex + bvhStart];

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

template<typename Scalar>
bool CollisionDetection<Scalar>::DetectCollisionCandidates(const BVHNode<Scalar>* dev_BVHNodes) {
    bool overflowHappened = false;
    bool overflow;
    dim3 numblocksTets = (mpSolverData->numTets + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(dev_numQueries, 0, sizeof(size_t));
    do {
        overflow = false;
        cudaMemset(dev_overflowFlag, 0, sizeof(bool));
        traverseTree << <numblocksTets, threadsPerBlock >> > (mpSolverData->numTets, dev_BVHNodes, mpSolverData->Tet, mpSolverData->dev_TetFathers, dev_queries, dev_numQueries, maxNumQueries, dev_overflowFlag);
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

template<typename Scalar>
void CollisionDetection<Scalar>::Init(int numTets, int numVerts, int maxThreads)
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

template<typename Scalar>
struct MinOp {
    __host__ __device__
        glm::tvec3<Scalar> operator()(const glm::tvec3<Scalar>& a, const glm::tvec3<Scalar>& b) const {
        return glm::min(a, b);
    }
};

template<typename Scalar>
struct MaxOp {
    __host__ __device__
        glm::tvec3<Scalar> operator()(const glm::tvec3<Scalar>& a, const glm::tvec3<Scalar>& b) const {
        return glm::max(a, b);
    }
};

template<typename Scalar>
AABB<Scalar> computeBoundingBox(const thrust::device_ptr<glm::tvec3<Scalar>>& begin, const thrust::device_ptr<glm::tvec3<Scalar>>& end) {
    glm::tvec3<Scalar> min = thrust::reduce(begin, end, glm::tvec3<Scalar>(FLT_MAX), MinOp<Scalar>());
    glm::tvec3<Scalar> max = thrust::reduce(begin, end, glm::tvec3<Scalar>(-FLT_MAX), MaxOp<Scalar>());

    return AABB<Scalar>{ min, max };
}

template<typename Scalar>
AABB<Scalar> CollisionDetection<Scalar>::GetAABB() const
{
    using XType = typename std::decay<decltype(*mpSolverData->X)>::type;
    thrust::device_ptr<XType> dev_ptr(mpSolverData->X);
    thrust::device_ptr<XType> dev_ptrTildes(mpSolverData->XTilde);
    return computeBoundingBox(dev_ptr, dev_ptr + numVerts).expand(computeBoundingBox(dev_ptrTildes, dev_ptrTildes + numVerts));
}

template<typename Scalar>
bool CollisionDetection<Scalar>::BroadPhase()
{
    const std::string buildTypeStr = buildType == BVH<Scalar>::BuildType::Cooperative ? "Cooperative" : buildType == BVH<Scalar>::BuildType::Atomic ? "Atomic" : "Serial";
    m_bvh.BuildBVHTree(buildType, GetAABB(), mpSolverData->numTets, mpSolverData->X, mpSolverData->XTilde, mpSolverData->Tet);

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

template class CollisionDetection<float>;
template class CollisionDetection<double>;