#pragma once

#include <def.h>
#include <simulation/solver/solverUtil.cuh>

namespace PdUtil {
    __global__ void CCDKernel(glm::vec3* X, glm::vec3* XTilde, glm::vec3* V, dataType* tI, glm::vec3* normal, float muT, float muN, int numVerts);

    __inline__ __device__ void setRowColVal(int index, int* idx, float* val, int r, int c, float v, int rowLen)
    {
        idx[index] = c * rowLen + r;
        val[index] = v;
    }

    __inline__ __device__ void setRowColVal(int index, int* rowIdx, int* colIdx, float* val, int r, int c, float v)
    {
        colIdx[index] = c;
        rowIdx[index] = r;
        val[index] = v;
    }

    __global__ void computeLocal(const float* V0, const float wi, float* xProj, const glm::mat3* DmInv, const float* qn__1, const indexType* tetIndex, int tetNumber);
    __global__ void computeSn(float* sn, float dt, float dt2_m_1, glm::vec3* pos, glm::vec3* vel, const glm::vec3* force, float* b, float massDt_2, int numVerts);
    __global__ void setMDt_2(int* rowIdx, int* colIdx, float* val, int startIndex, float massDt_2, int vertNumber);
    __global__ void computeM_h2Sn(float* b, float* sn, float massDt_2, int vertNumber);
    __global__ void addM_h2Sn(float* b, float* masses, int vertNumber);
    __global__ void computeSiTSi(int* rowIdx, int* colIdx, float* val, const float* V0, const glm::mat3* DmInv, const indexType* tetIndex, float weight, int tetNumber, int vertNumber);
    __global__ void updateVelPos(float* newPos, float dt_1, glm::vec3* pos, glm::vec3* vel, int numVerts);
}