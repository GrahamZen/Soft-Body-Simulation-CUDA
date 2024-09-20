#pragma once

#include <utilities.cuh>
#include <collision/bvh.cuh>
#include <collision/bvh.h>
#include <simulation/simulationContext.h>
#include <distance/distance_type.h>
#include <collision/intersections.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

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

template __device__ unsigned int genMortonCode<float>(AABB<float> bbox, glm::tvec3<float> geoMin, glm::tvec3<float> geoMax);
template __device__ unsigned int genMortonCode<double>(AABB<double> bbox, glm::tvec3<double> geoMin, glm::tvec3<double> geoMax);

__device__ unsigned long long expandMorton(int index, unsigned int mortonCode)
{
    unsigned long long exMortonCode = mortonCode;
    exMortonCode <<= 32;
    exMortonCode += index;
    return exMortonCode;
}

/**
* please sort the morton code first then get split pairs
thrust::stable_sort_by_key(mortonCodes, mortonCodes + TetrahedronCount, TriangleIndex);*/

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
__global__ void buildLeafMortonCCD(int startIndex, int numTri, Scalar minX, Scalar minY, Scalar minZ,
    Scalar maxX, Scalar maxY, Scalar maxZ, const indexType* tri, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, BVHNode<Scalar>* leafNodes,
    unsigned int* mortonCodes)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < numTri)
    {
        int leafPos = ind + numTri - 1;
        leafNodes[leafPos].bbox = computeTriTrajBBoxCCD(X[tri[ind * 3]], X[tri[ind * 3 + 1]], X[tri[ind * 3 + 2]],
            XTilde[tri[ind * 3]], XTilde[tri[ind * 3 + 1]], XTilde[tri[ind * 3 + 2]]);
        leafNodes[leafPos].isLeaf = 1;
        leafNodes[leafPos].leftIndex = -1;
        leafNodes[leafPos].rightIndex = -1;
        leafNodes[leafPos].TriangleIndex = ind;
        mortonCodes[ind + startIndex] = genMortonCode(leafNodes[ind + numTri - 1].bbox, glm::tvec3<Scalar>(minX, minY, minZ), glm::tvec3<Scalar>(maxX, maxY, maxZ));
    }
}

template<typename Scalar>
__global__ void buildLeafMorton(int startIndex, int numTri, Scalar minX, Scalar minY, Scalar minZ, Scalar maxX, Scalar maxY, Scalar maxZ,
    const indexType* tri, const glm::tvec3<Scalar>* X, BVHNode<Scalar>* leafNodes, unsigned int* mortonCodes, Scalar bound)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < numTri)
    {
        int leafPos = ind + numTri - 1;
        leafNodes[leafPos].bbox = computeTriTrajBBox(X[tri[ind * 3]], X[tri[ind * 3 + 1]], X[tri[ind * 3 + 2]], bound);
        leafNodes[leafPos].isLeaf = 1;
        leafNodes[leafPos].leftIndex = -1;
        leafNodes[leafPos].rightIndex = -1;
        leafNodes[leafPos].TriangleIndex = ind;
        mortonCodes[ind + startIndex] = genMortonCode(leafNodes[ind + numTri - 1].bbox, glm::tvec3<Scalar>(minX, minY, minZ), glm::tvec3<Scalar>(maxX, maxY, maxZ));
    }
}

template<typename Scalar>
void BVH<Scalar>::BuildBVHTreeCCD(BuildType buildType, const AABB<Scalar>& ctxAABB, int numTris, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, const indexType* tris)
{
    cudaMemset(dev_BVHNodes, 0, (numTris * 2 - 1) * sizeof(BVHNode<Scalar>));

    buildLeafMortonCCD << <numblocksTets, threadsPerBlock >> > (0, numTris, ctxAABB.min.x, ctxAABB.min.y, ctxAABB.min.z, ctxAABB.max.x, ctxAABB.max.y, ctxAABB.max.z,
        tris, X, XTilde, dev_BVHNodes, dev_mortonCodes);

    thrust::stable_sort_by_key(thrust::device, dev_mortonCodes, dev_mortonCodes + numTris, dev_BVHNodes + numTris - 1);

    buildSplitList << <numblocksTets, threadsPerBlock >> > (numTris, dev_mortonCodes, dev_BVHNodes);

    BuildBBoxes(buildType);

    cudaMemset(dev_mortonCodes, 0, numTris * sizeof(unsigned int));
    cudaMemset(dev_ready, 0, (numTris - 1) * sizeof(ReadyFlagType));
    cudaMemset(&dev_ready[numTris - 1], 1, numTris * sizeof(ReadyFlagType));
}

template<typename Scalar>
void BVH<Scalar>::BuildBVHTree(BuildType buildType, const AABB<Scalar>& ctxAABB, int numTris, const glm::tvec3<Scalar>* X, const indexType* tris, Scalar bound)
{
    cudaMemset(dev_BVHNodes, 0, (numTris * 2 - 1) * sizeof(BVHNode<Scalar>));

    buildLeafMorton << <numblocksTets, threadsPerBlock >> > (0, numTris, ctxAABB.min.x - bound, ctxAABB.min.y - bound, ctxAABB.min.z - bound,
        ctxAABB.max.x + bound, ctxAABB.max.y + bound, ctxAABB.max.z + bound,
        tris, X, dev_BVHNodes, dev_mortonCodes, bound);

    thrust::stable_sort_by_key(thrust::device, dev_mortonCodes, dev_mortonCodes + numTris, dev_BVHNodes + numTris - 1);

    buildSplitList << <numblocksTets, threadsPerBlock >> > (numTris, dev_mortonCodes, dev_BVHNodes);

    BuildBBoxes(buildType);

    cudaMemset(dev_mortonCodes, 0, numTris * sizeof(unsigned int));
    cudaMemset(dev_ready, 0, (numTris - 1) * sizeof(ReadyFlagType));
    cudaMemset(&dev_ready[numTris - 1], 1, numTris * sizeof(ReadyFlagType));
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

__constant__ int edgeIndicesTable[6] = {
    0, 1, 0, 2, 1, 2
};

template<typename T>
__host__ __device__ void sortThree(T& a, T& b, T& c) {
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
}

template __host__ __device__ void sortThree<indexType>(indexType& a, indexType& b, indexType& c);

template<typename T>
__host__ __device__ void sortFour(T& a, T& b, T& c, T& d) {
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (c > d) swap(c, d);
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
}

__device__ void fillQuery(Query* query, int triId, int tri2Id, const indexType* tris) {
    for (int i = 0; i < 3; i++) {
        int vi = tris[triId * 3 + i];
        int v20 = tris[tri2Id * 3 + 0];
        int v21 = tris[tri2Id * 3 + 1];
        int v22 = tris[tri2Id * 3 + 2];
        query[i].type = QueryType::VF;
        query[i].v0 = vi;
        query[i].v1 = v20;
        query[i].v2 = v21;
        query[i].v3 = v22;
    }
    for (int i = 0; i < 3; i++) {
        int v0 = tris[triId * 3 + edgeIndicesTable[i * 2 + 0]];
        int v1 = tris[triId * 3 + edgeIndicesTable[i * 2 + 1]];
        int v20 = tris[tri2Id * 3 + 0];
        int v21 = tris[tri2Id * 3 + 1];
        int v22 = tris[tri2Id * 3 + 2];
        query[i * 3 + 3].type = QueryType::EE;
        query[i * 3 + 3].v0 = v0;
        query[i * 3 + 3].v1 = v1;
        query[i * 3 + 3].v2 = v20;
        query[i * 3 + 3].v3 = v21;
        query[i * 3 + 4].type = QueryType::EE;
        query[i * 3 + 4].v0 = v0;
        query[i * 3 + 4].v1 = v1;
        query[i * 3 + 4].v2 = v20;
        query[i * 3 + 4].v3 = v22;
        query[i * 3 + 5].type = QueryType::EE;
        query[i * 3 + 5].v0 = v0;
        query[i * 3 + 5].v1 = v1;
        query[i * 3 + 5].v2 = v21;
        query[i * 3 + 5].v3 = v22;
    }
}

__device__ bool isAdjacentTriangle(indexType v00, indexType v01, indexType v02, indexType v10, indexType v11, indexType v12) {
    return (v00 == v10 || v00 == v11 || v00 == v12) ||
        (v01 == v10 || v01 == v11 || v01 == v12) ||
        (v02 == v10 || v02 == v11 || v02 == v12);
}

template<typename Scalar>
__global__ void traverseTree(int numTris, const BVHNode<Scalar>* nodes, const indexType* tris, const indexType* triFathers, Query* queries, size_t* queryCount, size_t maxNumQueries, bool* overflowFlag, bool ignoreSelfCollision)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numTris) return;
    int leafIdx = index + numTris - 1;
    const BVHNode<Scalar> myNode = nodes[leafIdx];
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
                // 1 faces * 3 verts + 3 edges * 3 edges
                if ((!ignoreSelfCollision || triFathers[myNode.TriangleIndex] != triFathers[leftChild.TriangleIndex]) && myNode.TriangleIndex != leftChild.TriangleIndex && !isAdjacentTriangle(tris[myNode.TriangleIndex * 3 + 0], tris[myNode.TriangleIndex * 3 + 1], tris[myNode.TriangleIndex * 3 + 2],
                    tris[leftChild.TriangleIndex * 3 + 0], tris[leftChild.TriangleIndex * 3 + 1], tris[leftChild.TriangleIndex * 3 + 2])) {
                    int qIdx = atomicAdd(queryCount, 12);
                    if (qIdx + 12 < maxNumQueries) {
                        Query* qBegin = &queries[qIdx];
                        fillQuery(qBegin, myNode.TriangleIndex, leftChild.TriangleIndex, tris);
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
                if ((!ignoreSelfCollision || triFathers[myNode.TriangleIndex] != triFathers[rightChild.TriangleIndex]) && myNode.TriangleIndex != rightChild.TriangleIndex && !isAdjacentTriangle(tris[myNode.TriangleIndex * 3 + 0], tris[myNode.TriangleIndex * 3 + 1], tris[myNode.TriangleIndex * 3 + 2],
                    tris[rightChild.TriangleIndex * 3 + 0], tris[rightChild.TriangleIndex * 3 + 1], tris[rightChild.TriangleIndex * 3 + 2])) {
                    int qIdx = atomicAdd(queryCount, 12);
                    if (qIdx + 12 < maxNumQueries) {
                        Query* qBegin = &queries[qIdx];
                        fillQuery(qBegin, myNode.TriangleIndex, rightChild.TriangleIndex, tris);
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
bool CollisionDetection<Scalar>::DetectCollisionCandidates(const BVHNode<Scalar>* dev_BVHNodes, int numTris, const indexType* Tri, const indexType* TriFathers) {
    bool overflowHappened = false;
    bool overflow;
    dim3 numblocksTets = (numTris + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(dev_numQueries, 0, sizeof(size_t));
    do {
        overflow = false;
        cudaMemset(dev_overflowFlag, 0, sizeof(bool));
        traverseTree << <numblocksTets, threadsPerBlock >> > (numTris, dev_BVHNodes, Tri, TriFathers, dev_queries, dev_numQueries, maxNumQueries, dev_overflowFlag, ignoreSelfCollision);
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
void CollisionDetection<Scalar>::Init(int numTris, int numVerts, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, int maxThreads)
{
    mpX = X;
    mpXTilde = XTilde;
    createQueries(numVerts);
    m_bvh.Init(numTris, numVerts, maxThreads);
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

template <typename Predicate>
void remove(Query* dev_queries, size_t& numQueries, Predicate pred) {
    thrust::device_ptr<Query> dev_ptr(dev_queries);

    auto new_end_remove = thrust::remove_if(dev_ptr, dev_ptr + numQueries, pred);
    numQueries = new_end_remove - dev_ptr;
}

void removeDuplicates(Query* dev_queries, size_t& numQueries) {
    thrust::device_ptr<Query> dev_ptr(dev_queries);

    auto new_end_remove = thrust::remove_if(dev_ptr, dev_ptr + numQueries, IsUnknown());
    numQueries = new_end_remove - dev_ptr;
    thrust::sort(dev_ptr, dev_ptr + numQueries, QueryComparator());

    auto new_end_unique = thrust::unique(dev_ptr, dev_ptr + numQueries, QueryEquality());

    numQueries = new_end_unique - dev_ptr;
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
AABB<Scalar> computeBoundingBox(const thrust::device_ptr<const glm::tvec3<Scalar>>& begin, const thrust::device_ptr<const glm::tvec3<Scalar>>& end) {
    glm::tvec3<Scalar> min = thrust::reduce(begin, end, glm::tvec3<Scalar>(FLT_MAX), MinOp<Scalar>());
    glm::tvec3<Scalar> max = thrust::reduce(begin, end, glm::tvec3<Scalar>(-FLT_MAX), MaxOp<Scalar>());

    return AABB<Scalar>{ min, max };
}

template<typename Scalar>
AABB<Scalar> GetAABBCCD(int numVerts, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde)
{
    thrust::device_ptr<const glm::tvec3<Scalar>> dev_ptr(X);
    thrust::device_ptr<const glm::tvec3<Scalar>> dev_ptrTildes(XTilde);
    return computeBoundingBox(dev_ptr, dev_ptr + numVerts).expand(computeBoundingBox(dev_ptrTildes, dev_ptrTildes + numVerts));
}

template<typename Scalar>
AABB<Scalar> GetAABB(int numVerts, const glm::tvec3<Scalar>* X)
{
    thrust::device_ptr<const glm::tvec3<Scalar>> dev_ptr(X);
    return computeBoundingBox(dev_ptr, dev_ptr + numVerts);
}

template<typename Scalar>
bool CollisionDetection<Scalar>::BroadPhaseCCD(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, const indexType* TriFathers)
{
    const std::string buildTypeStr = buildType == BVH<Scalar>::BuildType::Cooperative ? "Cooperative" : buildType == BVH<Scalar>::BuildType::Atomic ? "Atomic" : "Serial";
    m_bvh.BuildBVHTreeCCD(buildType, GetAABBCCD(numVerts, X, XTilde), numTris, X, XTilde, Tri);

    if (!DetectCollisionCandidates(m_bvh.GetBVHNodes(), numTris, Tri, TriFathers)) {
        count = 0;
        return false;
    }
    dim3 numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    sortEachQuery << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries);
    removeDuplicates(dev_queries, numQueries);
    count = numVerts;
    return true;
}

template<typename Scalar>
bool CollisionDetection<Scalar>::BroadPhase(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const indexType* TriFathers, Scalar bound)
{
    const std::string buildTypeStr = buildType == BVH<Scalar>::BuildType::Cooperative ? "Cooperative" : buildType == BVH<Scalar>::BuildType::Atomic ? "Atomic" : "Serial";
    m_bvh.BuildBVHTree(buildType, GetAABB(numVerts, X), numTris, X, Tri, bound);

    if (!DetectCollisionCandidates(m_bvh.GetBVHNodes(), numTris, Tri, TriFathers)) {
        count = 0;
        return false;
    }
    dim3 numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    sortEachQuery << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries);
    removeDuplicates(dev_queries, numQueries);
    count = numVerts;
    return true;
}

template<typename Scalar>
void CollisionDetection<Scalar>::UpdateQueries(int numVerts, int numTris, const indexType* Tri, const glm::tvec3<Scalar>* X, const indexType* TriFathers, Scalar dhat)
{
    BroadPhase(numVerts, numTris, Tri, X, TriFathers, dhat * 2);
    if (numQueries == 0)return;
    GetDistanceType<Scalar> << <(numQueries + 255) / 256, 256 >> > (X, dev_queries, numQueries);
    thrust::device_ptr<Query> queries_ptr(dev_queries);
    thrust::sort(queries_ptr, queries_ptr + numQueries, []__host__ __device__(const Query & a, const Query & b) { return a.dType < b.dType; });
    ComputeDistance<Scalar> << < (numQueries + 255) / 256, 256 >> > (X, dev_queries, numQueries);
    remove(dev_queries, numQueries, [dhat]__host__ __device__(const Query & q) { return q.d > dhat; });
    thrust::sort(queries_ptr, queries_ptr + numQueries, []__host__ __device__(const Query & a, const Query & b) {
        if (a.d == b.d) return a.dType < b.dType;
        return a.d < b.d;
    });
}

template class CollisionDetection<float>;
template class CollisionDetection<double>;