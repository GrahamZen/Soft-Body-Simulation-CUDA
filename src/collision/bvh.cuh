#pragma once
#include <crt/host_defines.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>
#include <vector>
#include <GL/glew.h>
#include <bvh.h>

AABB computeBoundingBox(const thrust::device_ptr<glm::vec3>& begin, const thrust::device_ptr<glm::vec3>& end);

__inline__ __device__ AABB computeBBox(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
{
    glm::vec3 min, max;
    min.x = fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x);
    min.y = fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y);
    min.z = fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z);
    max.x = fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x);
    max.y = fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y);
    max.z = fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z);

    return AABB{ min, max };
}

__inline__ __device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

//input the aabb box of a Tetrahedron
//generate a 30-bit morton code
__device__ unsigned int genMortonCode(AABB bbox, glm::vec3 geoMin, glm::vec3 geoMax);


__device__ unsigned long long expandMorton(int index, unsigned int mortonCode);

/**
* please sort the morton code first then get split pairs
thrust::stable_sort_by_key(mortonCodes, mortonCodes + TetrahedronCount, TetrahedronIndex);*/

//total input is a 30 x N matrix
//currentIndex is between 0 - N-1
//the input morton codes should be in the reduced form, no same elements are expected to appear twice!
__device__ int getSplit(unsigned int* mortonCodes, unsigned int currIndex, int nextIndex, unsigned int bound);

__inline__ __device__ int getSign(int tmp)
{
    if (tmp > 0)
        return 1;
    else
        return -1;
    //return (tmp > 0) - (tmp < 0);
}

__device__ void buildBBox(BVHNode& curr, BVHNode left, BVHNode right);

// build the bounding box and morton code for each Tetrahedron
__global__ void buildLeafMorton(int startIndex, int numTri, float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ, GLuint* tet, glm::vec3* X, BVHNode* leafNodes,
    unsigned int* mortonCodes);

//input the unique morton code
//codeCount is the size of the unique morton code
//splitList is 30 x N list
// the size of unique morton is less than 2^30 : [1, 2^30]
__global__ void buildSplitList(int codeCount, unsigned int* uniqueMorton, BVHNode* nodes);

// very naive implementation
__global__ void buildBBoxes(int leafCount, BVHNode* nodes, unsigned char* ready);

__device__ int traverseTree(const BVHNode* nodes, glm::vec3* X,
    int start, int end, AABB bbox, glm::vec3 X0, glm::vec3 dX0, int meshInd, int* indicesToReport);