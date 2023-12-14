#include <simulation/softBody.h>

SoftBody::~SoftBody()
{
    cudaFree(solverData.Tet);
    cudaFree(solverData.Force);
    cudaFree(solverData.V);
    cudaFree(solverData.inv_Dm);

    delete solver;
}

void SoftBody::Reset()
{
    cudaMemcpy(solverData.X, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.XTilt, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemset(solverData.V, 0, sizeof(glm::vec3) * solverData.numVerts);
}
