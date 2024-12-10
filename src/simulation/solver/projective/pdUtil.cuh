#pragma once

#include <def.h>
#include <simulation/solver/solverUtil.cuh>

namespace PdUtil {
    template<typename Scalar>
    __global__ void CCDKernel(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, glm::tvec3<Scalar>* V, Scalar* tI, glm::tvec3<Scalar>* normal, float muT, float muN, int numVerts);

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

    __global__ void computeLocal(const float* V0, const float* wi, float* xProj, const glm::mat3* DmInv, const float* qn__1, const indexType* tetIndex, int tetNumber, bool isJacobi = false);
    __global__ void computeDBCLocal(int numDBC, indexType* DBC, const glm::vec3* x0, const float wi, float* xProj);
    __global__ void computeSn(int numVerts, float* sn, float dt, const float* massDt_2s, glm::vec3* pos, glm::vec3* vel, const glm::vec3* force);
    __global__ void setMDt_2(int* rowIdx, int* colIdx, float* val, int offset, const float* masses, float dt2, float* massDt_2s, int numVerts);
    __global__ void setOne(int numDBC, indexType* DBC, int offset, int* rowIdx, int* colIdx, float* val, float weight);
    __global__ void computeM_h2Sn(float* b, float* sn, float massDt_2, int numVerts);
    __global__ void addM_h2Sn(float* b, float* sn, float* massDt_2s, int numVerts);
    __global__ void computeSiTSi(int* rowIdx, int* colIdx, float* val, const float* V0, const glm::mat3* DmInv, const indexType* tetIndex, const float* weight, int tetNumber, int numVerts);
    __global__ void updateVelPos(float* newPos, float dt_1, glm::vec3* pos, glm::vec3* vel, int numVerts);
}