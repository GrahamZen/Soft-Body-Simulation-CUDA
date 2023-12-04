#pragma once
#include <utilities.h>
#include <vector>

class SoftBody;
class Camera;
class SimulationCUDAContext;
class SurfaceShader;

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
    float Dt = 0.001;
    bool WireFrame = false;
    bool BVHVis = false;
    bool BVHEnabled = true;
    bool handleCollision = true;
    bool QueryVis = true;
    bool ObjectVis = true;
    bool Reset = false;
    bool Pause = false;
    bool Step = false;
    bool UseEigen = true;
    bool UseCUDASolver = true;
    float theta, phi;
    glm::vec3 cameraLookAt;
    float zoom;
    int currSimContextId = -1;
    struct SoftBodyAttr
    {
        int currSoftBodyId = -1;
        std::pair<float, bool> stiffness_0;
        std::pair<float, bool> stiffness_1;
        std::pair<float, bool> damp;
        std::pair<float, bool> muN;
        std::pair<float, bool> muT;
        void setJumpClean(bool& val);
        void setJump(bool val);
        bool getJumpDirty()const;
    private:
        std::pair<bool, bool> jump;
    }softBodyAttr;
};


void cleanupCuda();

class Context
{
    friend class SimulationCUDAContext;
public:
    Context(const std::string& _filename);
    ~Context();
    void LoadShaders(const std::string& vertShaderFilename = "../src/shaders/lambert.vert.glsl", const std::string& fragShaderFilename = "../src/shaders/lambert.frag.glsl");
    void LoadFlatShaders(const std::string& vertShaderFilename = "../src/shaders/flat.vert.glsl", const std::string& fragShaderFilename = "../src/shaders/flat.frag.glsl");
    void InitDataContainer();
    void InitCuda();
    void Update();
    void ResetCamera();
    void Draw();
    int GetNumQueries() const;
    int GetIteration() const { return iteration; }
    const std::vector<int>& GetDOFs() const { return DOFs; }
    const std::vector<int>& GetEles() const { return Eles; }
    const std::vector<const char*>& GetNamesSoftBodies() const;
    const std::vector<const char*>& GetNamesContexts() const { return namesContexts; }
    Camera* mpCamera = nullptr;
    const int width = 1024;
    const int height = 1024;
    bool panelModified = false;
    bool camchanged = false;

    float zoom, theta, phi;
    glm::vec3 cameraPosition;
    GuiDataContainer* guiData;
    SimulationCUDAContext* mcrpSimContext = nullptr;
    std::vector<SimulationCUDAContext*> mpSimContexts;

private:
    int GetMaxCGThreads();
    void PollEvents();
    std::string filename = "context.json";
    SimulationCUDAContext* LoadSimContext();
    glm::vec3 ogLookAt; // for recentering the camera
    SurfaceShader* mpProgLambert;
    SurfaceShader* mpProgFlat;
    int iteration = 0;
    bool pause = false;
    std::vector<const char*> namesContexts;
    std::vector<int> DOFs;
    std::vector<int> Eles;
};