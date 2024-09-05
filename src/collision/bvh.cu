#pragma once

#include <collision/bvh.h>
#include <utilities.cuh>
#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

template<typename Scalar>
__device__ void buildBBox(BVHNode<Scalar>& curr, const BVHNode<Scalar>& left, const BVHNode<Scalar>& right)
{
    glm::tvec3<Scalar> newMin;
    glm::tvec3<Scalar> newMax;
    newMin.x = glm::min(left.bbox.min.x, right.bbox.min.x);
    newMax.x = glm::max(left.bbox.max.x, right.bbox.max.x);
    newMin.y = glm::min(left.bbox.min.y, right.bbox.min.y);
    newMax.y = glm::max(left.bbox.max.y, right.bbox.max.y);
    newMin.z = glm::min(left.bbox.min.z, right.bbox.min.z);
    newMax.z = glm::max(left.bbox.max.z, right.bbox.max.z);

    curr.bbox = AABB<Scalar>{ newMin, newMax };
    curr.isLeaf = 0;
}

template<typename Scalar>
__global__ void buildBBoxesSerial(int leafCount, BVHNode<Scalar>* nodes, BVH<Scalar>::ReadyFlagType* ready) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= leafCount - 1)return;
    BVHNode<Scalar> node = nodes[ind];
    if (ready[ind] != 0)
        return;
    if (ready[node.leftIndex] != 0 && ready[node.rightIndex] != 0)
    {
        buildBBox(nodes[ind], nodes[node.leftIndex], nodes[node.rightIndex]);
        ready[ind] = 1;
    }
}

namespace cg = cooperative_groups;

template<typename Scalar>
__global__ void buildBBoxesCG(int leafCount, BVHNode<Scalar>* nodes, BVH<Scalar>::ReadyFlagType* ready) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    cg::grid_group grid = cg::this_grid();

    if (ind >= leafCount - 1)return;
    bool done = false;
    while (!done) {
        BVHNode<Scalar> node = nodes[ind];
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

template<typename Scalar>
__global__ void buildBBoxesAtomic(int leafCount, BVHNode<Scalar>* nodes, BVH<Scalar>::ReadyFlagType* ready) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= leafCount - 1) return;
    BVHNode<Scalar> node = nodes[ind];

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

template<typename Scalar>
void BVH<Scalar>::Init(int _numTris, int _numVerts, int maxThreads)
{
    numTris = _numTris;
    int numVerts = _numVerts;
    int numNodes = numTris * 2 - 1;
    cudaMalloc(&dev_BVHNodes, numNodes * sizeof(BVHNode<Scalar>));
    cudaMalloc((void**)&dev_tI, numVerts * sizeof(Scalar));
    cudaMemset(dev_tI, 0, numVerts * sizeof(Scalar));
    cudaMalloc((void**)&dev_indicesToReport, numVerts * sizeof(int));
    cudaMemset(dev_indicesToReport, -1, numVerts * sizeof(int));
    cudaMalloc(&dev_mortonCodes, numTris * sizeof(unsigned int));
    cudaMalloc(&dev_ready, numNodes * sizeof(ReadyFlagType));
    createBVH(numNodes);
    cudaMemset(dev_mortonCodes, 0, numTris * sizeof(unsigned int));
    cudaMemset(dev_ready, 0, (numTris - 1) * sizeof(ReadyFlagType));
    cudaMemset(&dev_ready[numTris - 1], 1, numTris * sizeof(ReadyFlagType));
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &suggestedBlocksize, buildBBoxesCG<Scalar>, 0, 0);

    if (numTris < maxThreads) {
        std::cout << "Using cooperative group." << std::endl;
        isBuildBBCG = true;
    }
    else {
        std::cout << "Not using cooperative group." << std::endl;
    }
    numblocksTets = (numTris + threadsPerBlock - 1) / threadsPerBlock;
    numblocksVerts = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    suggestedCGNumblocks = (numTris + suggestedBlocksize - 1) / suggestedBlocksize;
}

template<typename Scalar>
void BVH<Scalar>::BuildBBoxes(BuildType buildType) {
    if (buildType == BuildType::Cooperative && isBuildBBCG) {
        void* args[] = { &numTris, &dev_BVHNodes, &dev_ready };
        cudaError_t error = cudaLaunchCooperativeKernel((void*)buildBBoxesCG<Scalar>, suggestedCGNumblocks, suggestedBlocksize, args);
        if (error != cudaSuccess) {
            std::cerr << "cudaLaunchCooperativeKernel failed: " << cudaGetErrorString(error) << std::endl;
        }
    }
    else if (buildType == BuildType::Atomic) {
        buildBBoxesAtomic<Scalar> << < numblocksTets, threadsPerBlock >> > (numTris, dev_BVHNodes, dev_ready);
    }
    else if (buildType == BuildType::Serial) {
        ReadyFlagType treeBuild = 0;
        while (treeBuild == 0) {
            buildBBoxesSerial<Scalar> << < numblocksTets, threadsPerBlock >> > (numTris, dev_BVHNodes, dev_ready);
            cudaMemcpy(&treeBuild, dev_ready, sizeof(ReadyFlagType), cudaMemcpyDeviceToHost);
        }
    }
}

template<typename Scalar>
BVH<Scalar>::BVH<Scalar>(const int _threadsPerBlock) :
    threadsPerBlock(_threadsPerBlock) {}

template<typename Scalar>
BVH<Scalar>::~BVH<Scalar>()
{
    cudaFree(dev_BVHNodes);
    cudaFree(dev_tI);
    cudaFree(dev_indicesToReport);

    cudaFree(dev_ready);
    cudaFree(dev_mortonCodes);
}

template<typename Scalar>
void BVH<Scalar>::PrepareRenderData()
{
    glm::vec3* pos;
    Wireframe::MapDevicePosPtr(&pos);
    int numNodes = numTris * 2 - 1;
    dim3 numThreadsPerBlock(numNodes / threadsPerBlock + 1);
    populateBVHNodeAABBPos << <numThreadsPerBlock, threadsPerBlock >> > (dev_BVHNodes, pos, numNodes);
    Wireframe::UnMapDevicePtr();
}

template<typename Scalar>
const BVHNode<Scalar>* BVH<Scalar>::GetBVHNodes() const
{
    return dev_BVHNodes;
}

template<typename Scalar>
void CollisionDetection<Scalar>::DetectCollision(Scalar* tI, glm::vec3* nors)
{
    thrust::device_ptr<Scalar> dev_ptr(tI);
    thrust::fill(dev_ptr, dev_ptr + numVerts, 1.0f);
    if (BroadPhase()) {
        PrepareRenderData();
        NarrowPhase(tI, nors);
    }
}

template<typename Scalar>
void CollisionDetection<Scalar>::SetBuildType(int _buildType)
{
    buildType = static_cast<typename BVH<Scalar>::BuildType>(_buildType);
}

template<typename Scalar>
int CollisionDetection<Scalar>::GetBuildType()
{
    return static_cast<int>(buildType);
}

template class BVH<float>;
template class BVH<double>;

template class CollisionDetection<float>;
template class CollisionDetection<double>;