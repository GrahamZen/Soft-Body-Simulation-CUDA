#pragma once
#include <utilities.h>

class Camera;
class SoftBody;
class SimulationCUDAContext;
class SurfaceShader;

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
    float Dt = 0.001;
    bool WireFrame = false;
    bool Reset = false;
    float theta, phi;
    glm::vec3 cameraLookAt;
    float zoom;
    struct SoftBodyAttr
    {
        int currSoftBodyId = -1;
        std::pair<float, bool> stiffness_0;
        std::pair<float, bool> stiffness_1;
        std::pair<float, bool> damp;
        std::pair<float, bool> muN;
        std::pair<float, bool> muT;
        void setJump(bool val) { jump = { val,true }; }
        bool getJumpDirty()const { return jump.second; }
        bool getJumpState()const { return jump.first; }
        void cleanJump() { jump.second = false; }
    private:
        std::pair<bool, bool> jump;
    }softBodyAttr;
};


void cleanupCuda();

class Context
{
public:
    Context(const std::string& _filename);
    ~Context();
    void LoadShaders(const std::string& vertShaderFilename = "../src/shaders/lambert.vert.glsl", const std::string& fragShaderFilename = "../src/shaders/lambert.frag.glsl");
    void InitDataContainer();
    void InitCuda();
    void Update();
    void ResetCamera();
    void Draw();
    int GetIteration() const { return iteration; }
    const std::vector<const char*>& GetnamesSoftBodies() const { return namesSoftBodies; }
    Camera* mpCamera = nullptr;
    const int width = 1024;
    const int height = 1024;
    bool panelModified = false;
    bool camchanged = false;

    float zoom, theta, phi;
    glm::vec3 cameraPosition;
    GuiDataContainer* guiData;
    SimulationCUDAContext* mpSimContext;

private:
    void PollEvents();
    Camera* loadCamera(const std::string& _filename);
    std::string filename = "context.json";
    SimulationCUDAContext* LoadSimContext();
    glm::vec3 ogLookAt; // for recentering the camera
    SurfaceShader* mpProgLambert;
    int iteration = 0;
    std::vector<const char*> namesSoftBodies;
};