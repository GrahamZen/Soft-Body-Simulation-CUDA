#pragma once

#include <vector>
#include <mesh.h>
#include <context.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

class SimulationCUDAContext;

class SoftBody : public Mesh {
public:
    SoftBody(const char* nodeFileName, const char* eleFileName, SimulationCUDAContext*, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
        float mass = 1.0f, float stiffness_0 = 20000.0f, float stiffness_1 = 5000.0f, float damp = 0.999f, float muN = 0.5f, float muT = 0.5f, int constraints = 0,
        bool centralize = false, int startIndex = 0);
    ~SoftBody();

    void InitModel();
    void PdSolver();
    void Update();
    void Reset();
    void mapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr);
    void unMapDevicePtr();
    GLuint* getTet()const { return Tet; }
    glm::vec3* getX()const { return X; }
    glm::vec3* getV()const { return V; }
    glm::vec3* getForce()const { return Force; }
    glm::mat3* getInvDm()const { return inv_Dm; }
    void setAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr);
    int getNumber()const { return number; }
    int getTetNumber()const { return tet_number; }
    void Laplacian_Smoothing(float blendAlpha = 0.5f);
private:
    SimulationCUDAContext* mpSimContext;
    float mass = 1.0f;
    int numConstraints = 0;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    float damp = 0.999f;
    float muN = 0.5f;
    float muT = 0.5f;
    bool jump = false;

    GLuint* Tet;
    int tet_number; // The number of tetrahedra
    int number; // The number of vertices

    glm::vec3* Force;
    glm::vec3* V;
    glm::vec3* X;
    glm::vec3* X0;
    glm::vec3* Velocity;
    float* Mass;

    glm::mat3* inv_Dm;

    // For Laplacian smoothing.
    glm::vec3* V_sum;
    int* V_num;

    // Methods
    void _Update();
    void SetForce(Eigen::MatrixX3d* fext);
    std::vector<GLuint> loadEleFile(const std::string&, int startIndex = 0);
    std::vector<glm::vec3> loadNodeFile(const std::string&, bool centralize = false);
};
