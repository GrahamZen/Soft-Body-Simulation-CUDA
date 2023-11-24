#pragma once

#include <glm/glm.hpp>
#include <bvh.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <intersections.h>
#include <cuda_runtime.h>
#include <simulationContext.h>

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

//input the aabb box of a Tetrahedron
//generate a 30-bit morton code
__device__ unsigned int genMortonCode(AABB bbox, glm::vec3 geoMin, glm::vec3 geoMax)
{
    float x = (bbox.min.x + bbox.max.x) * 0.5f;
    float y = (bbox.min.y + bbox.max.y) * 0.5f;
    float z = (bbox.min.y + bbox.max.y) * 0.5f;
    float normalizedX = (x - geoMin.x) / (geoMax.x - geoMin.x);
    float normalizedY = (y - geoMin.y) / (geoMax.y - geoMin.y);
    float normalizedZ = (z - geoMin.z) / (geoMax.z - geoMin.z);

    normalizedX = glm::min(glm::max(normalizedX * 1024.0f, 0.0f), 1023.0f);
    normalizedY = glm::min(glm::max(normalizedY * 1024.0f, 0.0f), 1023.0f);
    normalizedZ = glm::min(glm::max(normalizedZ * 1024.0f, 0.0f), 1023.0f);

    unsigned int xx = expandBits((unsigned int)normalizedX);
    unsigned int yy = expandBits((unsigned int)normalizedY);
    unsigned int zz = expandBits((unsigned int)normalizedZ);

    return xx * 4 + yy * 2 + zz;
}


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

__device__ void buildBBox(BVHNode& curr, BVHNode left, BVHNode right)
{
    glm::vec3 newMin;
    glm::vec3 newMax;
    newMin.x = glm::min(left.bbox.min.x, right.bbox.min.x);
    newMax.x = glm::max(left.bbox.max.x, right.bbox.max.x);
    newMin.y = glm::min(left.bbox.min.y, right.bbox.min.y);
    newMax.y = glm::max(left.bbox.max.y, right.bbox.max.y);
    newMin.z = glm::min(left.bbox.min.z, right.bbox.min.z);
    newMax.z = glm::max(left.bbox.max.z, right.bbox.max.z);

    curr.bbox = AABB{ newMin, newMax };
    curr.isLeaf = 0;
}

// build the bounding box and morton code for each SoftBody
__global__ void buildLeafMorton(int startIndex, int numTri, float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ, GLuint* tet, glm::vec3* X, BVHNode* leafNodes,
    unsigned int* mortonCodes)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < numTri)
    {
        int leafPos = ind + numTri - 1;
        leafNodes[leafPos].bbox = computeBBox(X[tet[ind] * 4], X[tet[ind] * 4 + 1], X[tet[ind] * 4 + 2], X[tet[ind] * 4 + 3]);
        leafNodes[leafPos].isLeaf = 1;
        leafNodes[leafPos].leftIndex = -1;
        leafNodes[leafPos].rightIndex = -1;
        leafNodes[leafPos].TetrahedronIndex = ind;
        mortonCodes[ind + startIndex] = genMortonCode(leafNodes[ind + numTri - 1].bbox, glm::vec3(minX, minY, minZ), glm::vec3(maxX, maxY, maxZ));
    }
}


//input the unique morton code
//codeCount is the size of the unique morton code
//splitList is 30 x N list
// the size of unique morton is less than 2^30 : [1, 2^30]
__global__ void buildSplitList(int codeCount, unsigned int* uniqueMorton, BVHNode* nodes)
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


// very naive implementation
__global__ void buildBBoxes(int leafCount, BVHNode* nodes, unsigned char* ready)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    // only update internal node
    if (ind < leafCount - 1)
    {
        BVHNode node = nodes[ind];
        if (ready[ind] != 0)
            return;
        if (ready[node.leftIndex] != 0 && ready[node.rightIndex] != 0)
        {
            buildBBox(nodes[ind], nodes[node.leftIndex], nodes[node.rightIndex]);
            ready[ind] = 1;
        }
    }
}


__device__ float traverseTree(const BVHNode* nodes, glm::vec3* X,
    int start, int end, AABB bbox, glm::vec3 X0, glm::vec3 XTilt, int& hitTetId)
{
    // record the closest intersection
    float closest = FLT_MAX;
    glm::vec3 worldIntersect = glm::vec3(0.f);
    glm::vec3 objectIntersect = glm::vec3(0.f);

    if (!bboxIntersectionTest(X0, XTilt, bbox))
    {
        return -1;
    }
    //int bvhStart = 2 * start - geo.meshid;
    int bvhStart = 2 * start - 0;
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

        bool hitLeft = bboxIntersectionTest(X0, XTilt, leftChild.bbox);
        bool hitRight = bboxIntersectionTest(X0, XTilt, rightChild.bbox);
        if (hitLeft)
        {
            // check triangle intersection
            if (leftChild.isLeaf == 1)
            {
                const glm::vec3& v0 = X[leftChild.TetrahedronIndex * 4 + 0];
                const glm::vec3& v1 = X[leftChild.TetrahedronIndex * 4 + 1];
                const glm::vec3& v2 = X[leftChild.TetrahedronIndex * 4 + 2];
                const glm::vec3& v3 = X[leftChild.TetrahedronIndex * 4 + 3];
                float distance = tetrahedronIntersectionTest(X0, XTilt, v0, v1, v2, v3);
                // if is closer, then calculate normal and uv
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
                const glm::vec3& v0 = X[leftChild.TetrahedronIndex * 4 + 0];
                const glm::vec3& v1 = X[leftChild.TetrahedronIndex * 4 + 1];
                const glm::vec3& v2 = X[leftChild.TetrahedronIndex * 4 + 2];
                const glm::vec3& v3 = X[leftChild.TetrahedronIndex * 4 + 3];
                glm::vec3 tmpWorldIntersect = glm::vec3(0.f);
                glm::vec3 tmpObjectIntersect = glm::vec3(0.f);
                float distance = tetrahedronIntersectionTest(X0, XTilt, v0, v1, v2, v3);
                // if is closer, then calculate normal and uv
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


__global__ void detectCollisionCandidatesKern(int numVerts, const BVHNode* nodes, glm::vec3* X, glm::vec3 X0, glm::vec3 XTilt, int meshInd, int* indicesToReport, float* tI)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numVerts)
    {
        int hitTetId = -1;
        AABB bbox = computeBBox(X[index * 4], X[index * 4 + 1], X[index * 4 + 2], X[index * 4 + 3]);
        float distance = traverseTree(nodes, X, 0, 0, bbox, X0, XTilt, hitTetId);
        if (distance != -1)
        {
            indicesToReport[index] = hitTetId;
            tI[index] = distance;
        }
        else {
            tI[index] = 1;
            indicesToReport[index] = -1;
        }
    }
}

float* BVH::detectCollisionCandidates(GLuint* Tet, glm::vec3* X, glm::vec3* XTilt) const
{
    int blockSize1d = 128;
    dim3 numblocks = (numVerts + blockSize1d - 1) / blockSize1d;
    detectCollisionCandidatesKern << <numblocks, blockSize1d >> > (numVerts, dev_BVHNodes, X, glm::vec3(0.f), glm::vec3(0.f), 0, dev_indicesToReport, dev_tI);
    return dev_tI;
}

BVH::BVH(int& _threadsPerBlock) : threadsPerBlock(_threadsPerBlock) {}

BVH::~BVH()
{
    cudaFree(dev_BVHNodes);
    //cudaFree(dev_bboxes);
    cudaFree(dev_tI);
    cudaFree(dev_indicesToReport);
}

void BVH::Init(int numTets, int numSoftBodies, int numVerts)
{
    cudaMalloc(&dev_BVHNodes, (numTets * 2 - numSoftBodies) * sizeof(BVHNode));
    cudaMemset(dev_BVHNodes, 0, (numTets * 2 - numSoftBodies) * sizeof(BVHNode));
    //cudaMalloc(&dev_bboxes, bboxes.size() * sizeof(AABB));
    //cudaMemcpy(dev_bboxes, bboxes.data(), bboxes.size() * sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dev_tI, numVerts * sizeof(float));
    cudaMemset(dev_tI, 0, numVerts * sizeof(float));
    cudaMalloc((void**)&dev_indicesToReport, numVerts * sizeof(int));
    cudaMemset(dev_indicesToReport, -1, numVerts * sizeof(int));
}

void BVH::BuildBVHTree(int startIndexBVH, const AABB& ctxAABB, int triCount, const std::vector<SoftBody*>& softBodies)
{
    const int blockSize1d = 128;
    unsigned int* dev_mortonCodes = NULL;
    cudaMalloc((void**)&dev_mortonCodes, triCount * sizeof(unsigned int));
    cudaMemset(dev_mortonCodes, 0, triCount * sizeof(unsigned int));
    unsigned char* dev_ready = NULL;
    cudaMalloc((void**)&dev_ready, (triCount * 2 - 1) * sizeof(unsigned char));
    cudaMemset(dev_ready, 0, triCount * sizeof(unsigned char));
    cudaMemset(&dev_ready[triCount - 1], 1, triCount * sizeof(unsigned char));

    static BVHNode* dev_tmpBVHNodes = NULL;
    cudaMalloc((void**)&dev_tmpBVHNodes, (triCount * 2 - 1) * sizeof(BVHNode));
    cudaMemset(dev_tmpBVHNodes, 0, (triCount * 2 - 1) * sizeof(BVHNode));

    dim3 numblocks = (triCount + blockSize1d - 1) / blockSize1d;
    int startIndexTri = 0;
    for (auto softBody : softBodies) {
        buildLeafMorton << <numblocks, blockSize1d >> > (startIndexTri, triCount, ctxAABB.min.x, ctxAABB.min.y, ctxAABB.min.z, ctxAABB.max.x, ctxAABB.max.y, ctxAABB.max.z,
            softBody->getTet(), softBody->getX(), dev_tmpBVHNodes, dev_mortonCodes);
        startIndexTri += softBody->getTetNumber();
    }

    thrust::stable_sort_by_key(dev_mortonCodes, dev_mortonCodes + triCount, dev_tmpBVHNodes + triCount - 1);


    /*
    unsigned int* hstMorton = (unsigned int*)malloc(sizeof(unsigned int) * triCount);
    cudaMemcpy(hstMorton, dev_mortonCodes, triCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 20; i++)
    {
        cout << std::bitset<30>(hstMorton[i]) << endl;
    }
    cout << endl;
    free(hstMorton);*/



    buildSplitList << <numblocks, blockSize1d >> > (triCount, dev_mortonCodes, dev_tmpBVHNodes);

    //can use atomic operation for further optimization
    for (int i = 0; i < triCount; i++)
    {
        buildBBoxes << <numblocks, blockSize1d >> > (triCount, dev_tmpBVHNodes, dev_ready);
    }

    cudaMemcpy(dev_BVHNodes + startIndexBVH, dev_tmpBVHNodes, (triCount * 2 - 1) * sizeof(BVHNode), cudaMemcpyDeviceToDevice);

    /*
    BVHNode* hstBVHNodes = (BVHNode*)malloc(sizeof(BVHNode) * (startIndexBVH + 2 * triCount - 1));
    cudaMemcpy(hstBVHNodes, dev_BVHNodes, sizeof(BVHNode) * (startIndexBVH + 2 * triCount - 1), cudaMemcpyDeviceToHost);
    for (int i = 0; i < startIndexBVH + 2 * triCount - 1; i++)
    {
        cout << i << ": " << hstBVHNodes[i].leftIndex << "," << hstBVHNodes[i].rightIndex << "  parent:" << hstBVHNodes[i].parent << endl;
        cout << i << ": " << hstBVHNodes[i].bbox.max.x << "," << hstBVHNodes[i].bbox.max.y << "," << hstBVHNodes[i].bbox.max.z << endl;
        //cout << i << ": " << hstBVHNodes[i].bbox.min.x << "," << hstBVHNodes[i].bbox.min.y << "," << hstBVHNodes[i].bbox.min.z << endl;
    }
    cout << endl;
    cout << endl;
    cout << endl;
    free(hstBVHNodes);*/


    cudaFree(dev_ready);
    cudaFree(dev_mortonCodes);
    cudaFree(dev_tmpBVHNodes);
}