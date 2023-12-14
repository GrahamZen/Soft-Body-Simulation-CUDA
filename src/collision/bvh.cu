#pragma once

#include <glm/glm.hpp>
#include <bvh.cuh>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <utilities.cuh>
#include <simulationContext.h>

//input the aabb box of a Tetrahedron
//generate a 30-bit morton code
__device__ unsigned int genMortonCode(AABB bbox, glmVec3 geoMin, glmVec3 geoMax)
{
    dataType x = (bbox.min.x + bbox.max.x) * 0.5f;
    dataType y = (bbox.min.y + bbox.max.y) * 0.5f;
    dataType z = (bbox.min.y + bbox.max.y) * 0.5f;
    dataType normalizedX = (x - geoMin.x) / (geoMax.x - geoMin.x);
    dataType normalizedY = (y - geoMin.y) / (geoMax.y - geoMin.y);
    dataType normalizedZ = (z - geoMin.z) / (geoMax.z - geoMin.z);

    normalizedX = glm::min(glm::max(normalizedX * 1024.0, 0.0), 1023.0);
    normalizedY = glm::min(glm::max(normalizedY * 1024.0, 0.0), 1023.0);
    normalizedZ = glm::min(glm::max(normalizedZ * 1024.0, 0.0), 1023.0);

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

__device__ void buildBBox(BVHNode& curr, const BVHNode& left, const BVHNode& right)
{
    glmVec3 newMin;
    glmVec3 newMax;
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
__global__ void buildLeafMorton(int startIndex, int numTri, dataType minX, dataType minY, dataType minZ,
    dataType maxX, dataType maxY, dataType maxZ, const GLuint* tet, const glm::vec3* X, const glm::vec3* XTilt, BVHNode* leafNodes,
    unsigned int* mortonCodes)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < numTri)
    {
        int leafPos = ind + numTri - 1;
        leafNodes[leafPos].bbox = computeTetTrajBBox(X[tet[ind * 4]], X[tet[ind * 4 + 1]], X[tet[ind * 4 + 2]], X[tet[ind * 4 + 3]],
            XTilt[tet[ind * 4]], XTilt[tet[ind * 4 + 1]], XTilt[tet[ind * 4 + 2]], XTilt[tet[ind * 4 + 3]]);
        leafNodes[leafPos].isLeaf = 1;
        leafNodes[leafPos].leftIndex = -1;
        leafNodes[leafPos].rightIndex = -1;
        leafNodes[leafPos].TetrahedronIndex = ind;
        mortonCodes[ind + startIndex] = genMortonCode(leafNodes[ind + numTri - 1].bbox, glmVec3(minX, minY, minZ), glmVec3(maxX, maxY, maxZ));
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

__global__ void buildBBoxesSerial(int leafCount, BVHNode* nodes, BVH::ReadyFlagType* ready) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= leafCount - 1)return;
    BVHNode node = nodes[ind];
    if (ready[ind] != 0)
        return;
    if (ready[node.leftIndex] != 0 && ready[node.rightIndex] != 0)
    {
        buildBBox(nodes[ind], nodes[node.leftIndex], nodes[node.rightIndex]);
        ready[ind] = 1;
    }
}

namespace cg = cooperative_groups;

__global__ void buildBBoxesCG(int leafCount, BVHNode* nodes, BVH::ReadyFlagType* ready) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    cg::grid_group grid = cg::this_grid();

    if (ind >= leafCount - 1)return;
    bool done = false;
    while (!done) {
        BVHNode node = nodes[ind];
        if (ready[ind] != 0) {}
        else if (ready[node.leftIndex] != 0 && ready[node.rightIndex] != 0)
        {
            buildBBox(nodes[ind], nodes[node.leftIndex], nodes[node.rightIndex]);
            ready[ind] = 1;
        }
        cg::sync(grid);
        done = ready[0] == 1;
        cg::sync(grid);
    }
}

__global__ void buildBBoxesAtomic(int leafCount, BVHNode* nodes, BVH::ReadyFlagType* ready) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= leafCount - 1) return;
    BVHNode node = nodes[ind];

    while (true) {
        auto leftReady = atomicCAS(&ready[node.leftIndex], 0, 0);
        auto rightReady = atomicCAS(&ready[node.rightIndex], 0, 0);

        if (leftReady != 0 && rightReady != 0) {
            buildBBox(nodes[ind], nodes[node.leftIndex], nodes[node.rightIndex]);
            ready[ind] = 1;
            break;
        }
        __threadfence();
    }
}


void BVH::Init(int _numTets, int _numVerts, int maxThreads)
{
    numTets = _numTets;
    numVerts = _numVerts;
    numNodes = numTets * 2 - 1;
    cudaMalloc(&dev_BVHNodes, numNodes * sizeof(BVHNode));
    cudaMalloc((void**)&dev_tI, numVerts * sizeof(dataType));
    cudaMemset(dev_tI, 0, numVerts * sizeof(dataType));
    cudaMalloc((void**)&dev_indicesToReport, numVerts * sizeof(int));
    cudaMemset(dev_indicesToReport, -1, numVerts * sizeof(int));
    cudaMalloc(&dev_mortonCodes, numTets * sizeof(unsigned int));
    cudaMalloc(&dev_ready, numNodes * sizeof(ReadyFlagType));
    createBVH(numNodes);
    cudaMemset(dev_mortonCodes, 0, numTets * sizeof(unsigned int));
    cudaMemset(dev_ready, 0, (numTets - 1) * sizeof(ReadyFlagType));
    cudaMemset(&dev_ready[numTets - 1], 1, numTets * sizeof(ReadyFlagType));
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &suggestedBlocksize, buildBBoxesCG, 0, 0);

    if (numTets < maxThreads) {
        std::cout << "Using cooperative group." << std::endl;
        isBuildBBCG = true;
    }
    else {
        std::cout << "Not using cooperative group." << std::endl;
    }
    numblocksTets = (numTets + threadsPerBlock - 1) / threadsPerBlock;
    numblocksVerts = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    suggestedCGNumblocks = (numTets + suggestedBlocksize - 1) / suggestedBlocksize;
}

void BVH::BuildBBoxes() {
    if (buildType == BuildType::Cooperative && isBuildBBCG) {
        void* args[] = { &numTets, &dev_BVHNodes, &dev_ready };
        cudaError_t error = cudaLaunchCooperativeKernel((void*)buildBBoxesCG, suggestedCGNumblocks, suggestedBlocksize, args);
        if (error != cudaSuccess) {
            std::cerr << "cudaLaunchCooperativeKernel failed: " << cudaGetErrorString(error) << std::endl;
        }
    }
    else if (buildType == BuildType::Atomic) {
        buildBBoxesAtomic << < numblocksTets, threadsPerBlock >> > (numTets, dev_BVHNodes, dev_ready);
    }
    else if (buildType == BuildType::Serial) {
        ReadyFlagType treeBuild = 0;
        while (treeBuild == 0) {
            buildBBoxesSerial << < numblocksTets, threadsPerBlock >> > (numTets, dev_BVHNodes, dev_ready);
            cudaMemcpy(&treeBuild, dev_ready, sizeof(ReadyFlagType), cudaMemcpyDeviceToHost);
        }
    }
}

void BVH::BuildBVHTree(const AABB& ctxAABB, int numTets, const glm::vec3* X, const glm::vec3* XTilt, const GLuint* tets)
{
    cudaMemset(dev_BVHNodes, 0, (numTets * 2 - 1) * sizeof(BVHNode));

    buildLeafMorton << <numblocksTets, threadsPerBlock >> > (0, numTets, ctxAABB.min.x, ctxAABB.min.y, ctxAABB.min.z, ctxAABB.max.x, ctxAABB.max.y, ctxAABB.max.z,
        tets, X, XTilt, dev_BVHNodes, dev_mortonCodes);

    thrust::stable_sort_by_key(thrust::device, dev_mortonCodes, dev_mortonCodes + numTets, dev_BVHNodes + numTets - 1);

    buildSplitList << <numblocksTets, threadsPerBlock >> > (numTets, dev_mortonCodes, dev_BVHNodes);

    BuildBBoxes();

    cudaMemset(dev_mortonCodes, 0, numTets * sizeof(unsigned int));
    cudaMemset(dev_ready, 0, (numTets - 1) * sizeof(ReadyFlagType));
    cudaMemset(&dev_ready[numTets - 1], 1, numTets * sizeof(ReadyFlagType));
}

BVH::BVH(const int _threadsPerBlock) :
    threadsPerBlock(_threadsPerBlock), buildType(BuildType::Serial) {}

BVH::~BVH()
{
    cudaFree(dev_BVHNodes);
    cudaFree(dev_tI);
    cudaFree(dev_indicesToReport);

    cudaFree(dev_ready);
    cudaFree(dev_mortonCodes);
}

void BVH::PrepareRenderData()
{
    glm::vec3* pos;
    Wireframe::MapDevicePosPtr(&pos);
    dim3 numThreadsPerBlock(numNodes / threadsPerBlock + 1);
    populateBVHNodeAABBPos << <numThreadsPerBlock, threadsPerBlock >> > (dev_BVHNodes, pos, numNodes);
    Wireframe::unMapDevicePtr();
}

const BVHNode* BVH::GetBVHNodes() const
{
    return dev_BVHNodes;
}

void CollisionDetection::DetectCollision(dataType* tI, glm::vec3* nors)
{
    thrust::device_ptr<dataType> dev_ptr(tI);
    thrust::fill(dev_ptr, dev_ptr + numVerts, 1.0f);
    if (BroadPhase(mPSimContext->numTets, mPSimContext->dev_Tets, mPSimContext->dev_TetFathers)) {
        PrepareRenderData(mPSimContext->dev_Xs);
        NarrowPhase(mPSimContext->dev_Xs, mPSimContext->dev_XTilts, tI, nors);
    }
}

void BVH::SetBuildType(BuildType _buildType)
{
    buildType = _buildType;
}

BVH::BuildType BVH::GetBuildType()
{
    return buildType;
}
