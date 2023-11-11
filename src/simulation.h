#pragma once

#include <vector>
#include "scene.h"
#include "mesh.h"

class SoftBody : public Mesh {
public:
    SoftBody(const char* nodeFileName, const char* eleFileName);
    ~SoftBody();

    void Update();
    void mapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr);
    void unMapDevicePtr();
    GLuint* getTet()const { return Tet; }
    glm::vec3* getX()const { return X; }
    glm::vec3* getV()const { return V; }
    glm::vec3* getForce()const { return Force; }
    glm::mat4* getInvDm()const { return inv_Dm; }
    int getNumber()const { return number; }
    int getTetNumber()const { return tet_number; }
private:
    float dt = 0.003f;
    float mass = 1.0f;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    float damp = 0.999f;
    float muN = 0.5f;
    float muT = 0.5f;
    bool neoHookean = false;

    GLuint* Tet;
    int tet_number; // The number of tetrahedra
    int number; // The number of vertices

    glm::vec3* Force;
    glm::vec3* V;
    glm::vec3* X;

    glm::mat4* inv_Dm;

    // For Laplacian smoothing.
    glm::vec3* V_sum;
    int* V_num;

    // Methods
    void _Update();
    glm::mat4 Build_Edge_Matrix(int tet);
    std::vector<GLuint> loadEleFile(const std::string&);
    std::vector<glm::vec3> loadNodeFile(const std::string&);
};

class SimulationCUDAContext {
public:
    SimulationCUDAContext();
    ~SimulationCUDAContext();
    void Update();
    std::vector<SoftBody*> softBodies;
};



void InitDataContainer(GuiDataContainer* guiData);