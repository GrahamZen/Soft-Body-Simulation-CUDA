#pragma once
#include <def.h>
#include <precision.h>
#include <vector>
#include <string>

class SoftBody;
class Camera;
class SimulationCUDAContext;
class SurfaceShader;
class TextureCubemap;
class Mesh;


struct SoftBodyAttr
{
    int currSoftBodyId = -1;
    bool mu;
    bool lambda;
    bool damp;
    bool muN;
    bool muT;
    void setJumpClean(bool& val);
    void setJump(bool val);
    bool getJumpDirty()const;
private:
    std::pair<bool, bool> jump;
};

class GuiDataContainer
{
public:
    GuiDataContainer();
    ~GuiDataContainer();
    SolverParams<solverPrecision>* solverParams = nullptr;
    size_t PauseIter = (size_t)-1;
    float PointSize = 5;
    float LineWidth = 1;
    bool WireFrame = false;
    bool BVHVis = false;
    bool BVHEnabled = true;
    bool PerfEnabled = true;
    bool handleCollision = true;
    bool QueryVis = false;
    bool QueryDebugMode = true;
    bool ObjectVis = true;
    bool Reset = false;
    bool Pause = false;
    bool Step = false;
    bool UseEigen = true;
    int CurrQueryId = 0;
    std::string HighLightObjId;
    float theta, phi;
    glm::vec3 cameraLookAt;
    float zoom;
    int currSimContextId = -1;
    Query* mPQuery;
    bool QueryDirty = true;
    SoftBodyAttr softBodyAttr;
};


void cleanupCuda();

class Context
{
public:
    Context(const std::string& _filename);
    ~Context();
    void LoadShaders(const std::string& vertShaderFilename = "../src/shaders/lambert.vert.glsl", const std::string& fragShaderFilename = "../src/shaders/lambert.frag.glsl");
    void LoadFlatShaders(const std::string& vertShaderFilename = "../src/shaders/flat.vert.glsl", const std::string& fragShaderFilename = "../src/shaders/flat.frag.glsl");
    void LoadEnvCubemap(const std::string& filename);
    void InitDataContainer();
    void InitCuda();
    void Update();
    void ResetCamera();
    void Draw();
    void SetBVHBuildType(int buildType);
    int& GetBVHBuildType();
    int GetNumQueries() const;
    int GetIteration() const { return iteration; }
    const std::vector<int>& GetDOFs() const { return DOFs; }
    const std::vector<int>& GetEles() const { return Eles; }
    Camera* mpCamera = nullptr;
    const int width = 1024;
    const int height = 1024;
    bool panelModified = false;
    bool camchanged = false;
    int bvhBuildType = 1;
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
    SurfaceShader* mpProgHighLight = nullptr;
    SurfaceShader* mpProgLambert = nullptr;
    SurfaceShader* mpProgFlat = nullptr;
    SurfaceShader* mpProgSkybox = nullptr;
    Mesh* mpCube = nullptr;
    size_t iteration = 0;
    bool pause = false;
    bool logEnabled = false;
    std::vector<int> DOFs;
    std::vector<int> Eles;
    TextureCubemap* envMap = nullptr;
};