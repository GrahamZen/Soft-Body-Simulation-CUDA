#pragma once

#include <def.h>
#include <vector>

template<typename Scalar>
__device__ AABB<Scalar> computeTetTrajBBox(const glm::tvec3<Scalar>& v0, const glm::tvec3<Scalar>& v1, const glm::tvec3<Scalar>& v2, const glm::tvec3<Scalar>& v3,
    const glm::tvec3<Scalar>& v4, const glm::tvec3<Scalar>& v5, const glm::tvec3<Scalar>& v6, const glm::tvec3<Scalar>& v7);

template<typename Scalar>
__device__ AABB<Scalar> computeTriTrajBBox(const glm::tvec3<Scalar>& v0, const glm::tvec3<Scalar>& v1, const glm::tvec3<Scalar>& v2, const glm::tvec3<Scalar>& v3,
    const glm::tvec3<Scalar>& v4, const glm::tvec3<Scalar>& v5);

template<typename Scalar>
__device__ unsigned int genMortonCode(AABB<Scalar> bbox, glm::tvec3<Scalar> geoMin, glm::tvec3<Scalar> geoMax);

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
template<typename Scalar>
__global__ void buildSplitList(int codeCount, unsigned int* uniqueMorton, BVHNode<Scalar>* nodes);