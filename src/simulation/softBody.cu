#include <simulation/softBody.h>

SoftBody::~SoftBody()
{
    if (solverData.inv_Dm)
        cudaFree(solverData.inv_Dm);
    if (solverData.dev_ExtForce)
        cudaFree(solverData.dev_ExtForce);
    if (solverData.V0)
        cudaFree(solverData.V0);
}

void SoftBody::Reset()
{
    cudaMemcpy(solverData.X, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.XTilt, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemset(solverData.V, 0, sizeof(glm::vec3) * solverData.numVerts);
}
