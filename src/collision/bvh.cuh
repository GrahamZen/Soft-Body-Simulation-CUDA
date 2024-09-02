#pragma once

#include <def.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <vector>

template<typename HighP>
__device__ AABB<HighP> computeTetTrajBBox(const glm::tvec3<HighP>& v0, const glm::tvec3<HighP>& v1, const glm::tvec3<HighP>& v2, const glm::tvec3<HighP>& v3,
    const glm::tvec3<HighP>& v4, const glm::tvec3<HighP>& v5, const glm::tvec3<HighP>& v6, const glm::tvec3<HighP>& v7);

template<typename HighP>
__device__ unsigned int genMortonCode(AABB<HighP> bbox, glm::tvec3<HighP> geoMin, glm::tvec3<HighP> geoMax);

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
template<typename HighP>
__global__ void buildSplitList(int codeCount, unsigned int* uniqueMorton, BVHNode<HighP>* nodes);