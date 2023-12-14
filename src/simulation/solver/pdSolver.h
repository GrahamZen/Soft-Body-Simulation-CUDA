#pragma once
#include <femSolver.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <thrust/device_vector.h>
#include <context.h>

class SimulationCUDAContext;
class PdSolver : public FEMSolver {
public:
    PdSolver(SimulationCUDAContext*, SolverAttribute&, const SolverData&);
    ~PdSolver();
    virtual void Update(SolverData& solverData) override;
protected:
    virtual void SolverPrepare(SolverData& solverData) override;
    virtual void SolverStep(SolverData& solverData) override;
    void setAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void Laplacian_Smoothing(float blendAlpha = 0.5f);
private:
    bool jump = false;

    indexType* Tri;

    int numTris; // The number of triangles
    int nnzNumber;

    bool solverReady = false;

    float* Mass;
    float* V0;

    csrcholInfo_t d_info = NULL;
    void* buffer_gpu = NULL;
    cusolverSpHandle_t cusolverHandle;

    float* masses;
    float* sn;
    float* b;


    // For Laplacian smoothing.
    glm::vec3* V_sum;
    int* V_num;

    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;

    // Methods
    void _Update();
    void SetForce(Eigen::MatrixX3d* fext);
};
