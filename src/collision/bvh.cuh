#pragma once
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>
#include <vector>
#include <GL/glew.h>
#include <bvh.h>

AABB computeBoundingBox(const thrust::device_ptr<glm::vec3>& begin, const thrust::device_ptr<glm::vec3>& end);

__device__ AABB computeTetTrajBBox(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3,
    const glm::vec3& v4, const glm::vec3& v5, const glm::vec3& v6, const glm::vec3& v7);

__inline__ __device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


__inline__ __device__ int getSign(int tmp)
{
    if (tmp > 0)
        return 1;
    else
        return -1;
    //return (tmp > 0) - (tmp < 0);
}


__device__ void traverseTree(const BVHNode* nodes, const glm::vec3* Xs, const glm::vec3* XTilts, int tetId, int* hitTetId, int& numHitTet);