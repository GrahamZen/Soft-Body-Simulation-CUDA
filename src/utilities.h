#pragma once

#include <glm/glm.hpp>
#include <string>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

class Camera;
class SoftBody;
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
    SoftBody* currSoftBody = nullptr;
};

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}

template <typename T>
void inspectHost(T*, int);
void inspectHost(unsigned int*, int);

class SimulationCUDAContext;
class SurfaceShader;

void cleanupCuda();

class Context {
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
    std::string filename = "context.json";
    SimulationCUDAContext* LoadSimContext();
    glm::vec3 ogLookAt; // for recentering the camera
    Camera* loadCamera(const std::string& _filename);
    SurfaceShader* mpProgLambert;
    int iteration = 0;
};