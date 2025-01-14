#pragma once

#include <def.h>
#include <simulation/solver/solverUtil.cuh>

namespace PdUtil {
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

    __global__ void computeSiTSi(int* rowIdx, int* colIdx, float* val, float* matrix_diag, const float* V0, const glm::mat3* DmInv, const indexType* tetIndex, const float* weight, int tetNumber, int numVerts);
    __global__ void setMDt_2(int numVerts, int* rowIdx, int* colIdx, float* val, int offset, const float* masses, float dt2, float* massDt_2s, float* DBC, float weight);
    __global__ void setMDt_2MoreDBC(int numVerts, const float* masses, float dt2, float* massDt_2s, float* moreDBC, float* DBC);
    __global__ void computeLocal(const float* V0, const float* wi, float* xProj, const glm::mat3* DmInv, const float* qn__1, const indexType* tetIndex, int tetNumber, bool isJacobi = false);
    __global__ void computeDBCLocal(int numVerts, float* DBC, float* moreDBC, const glm::vec3* x0, const float wi, float* xProj);
    __global__ void computeSn(int numVerts, float* sn, float dt, const float* massDt_2s, glm::vec3* pos, glm::vec3* vel, const glm::vec3* force, const float* more_fixed, const glm::vec3* offset_X, glm::vec3* fixed_X, glm::vec3 dir);
    __global__ void addM_h2Sn(float* b, float* sn, float* massDt_2s, int numVerts);
    __global__ void updateVelPos(float* newPos, float dt_1, glm::vec3* pos, glm::vec3* vel, int numVerts);

    // Jacobi
    __global__ void getErrorKern(int numVerts, float* next_x, const float* b, const float* massDt_2s, const float* sn, const float* matrix_diag);
    __global__ void chebyshevKern(int numVerts3, float* next_x, float* prev_x, float* sn, float omega);
}