#include <context.h>
#include <sceneStructs.h>
#include <surfaceshader.h>
#include <simulationContext.h>
#include <iostream>
#include <fstream>
#include <sphere.h>
#include <cylinder.h>
#include <plane.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

std::string getCurrentTimeStamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm bt = *std::localtime(&in_time_t);

    std::ostringstream ss;
    ss << std::put_time(&bt, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

Camera::Camera(nlohmann::json& camJson)
{
    resolution.y = camJson["screen height"];
    float aspectRatio = camJson["aspect ratio"];
    resolution.x = aspectRatio * resolution.y;
    position = glm::vec3(camJson["position"][0],
        camJson["position"][1],
        camJson["position"][2]);
    lookAt = glm::vec3(camJson["lookAt"][0],
        camJson["lookAt"][1],
        camJson["lookAt"][2]);
    view = glm::vec3(camJson["view"][0],
        camJson["view"][1],
        camJson["view"][2]);
    up = glm::vec3(camJson["up"][0],
        camJson["up"][1],
        camJson["up"][2]);
    fov.y = camJson["fovy"];
    computeCameraParams();
}

Camera::Camera(const std::string& _filename)
{
    std::ifstream fileStream = utilityCore::findFile(_filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << _filename << std::endl;
    }
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();
    if (json.contains("default camera")) {
        *this = Camera(json["default camera"]);
    }
}

Camera& Camera::computeCameraParams()
{
    // assuming resolution, position, lookAt, view, up, fovy are already set
    float yscaled = tan(fov.y * (PI / 180));
    float xscaled = (yscaled * resolution.x) / resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    fov.x = fovx;

    right = glm::normalize(glm::cross(view, up));
    pixelLength = glm::vec2(2 * xscaled / (float)resolution.x, 2 * yscaled / (float)resolution.y);
    return *this;
}

Context::Context(const std::string& _filename) :filename(_filename), mpCamera(new Camera(_filename)), mpProgLambert(new SurfaceShader()), mpProgFlat(new SurfaceShader()),
width(mpCamera->resolution.x), height(mpCamera->resolution.y), ogLookAt(mpCamera->lookAt), guiData(new GuiDataContainer())
{
    glm::vec3 view = mpCamera->view;
    glm::vec3 up = mpCamera->up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = mpCamera->position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    zoom = glm::length(mpCamera->position - ogLookAt);
    std::string filename = "logs/log_" + getCurrentTimeStamp() + ".txt";
    auto logger = spdlog::basic_logger_mt("basic_logger", filename);
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);
}

Context::~Context()
{
    for (auto name : namesContexts) {
        delete[]name;
    }
    delete mpProgLambert;
    delete mpProgFlat;
    delete mcrpSimContext;
    delete guiData;
    delete mpCamera;
}

int Context::GetMaxCGThreads()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int maxThreadsPerSM = props.maxThreadsPerMultiProcessor;
    int numSMs = props.multiProcessorCount;
    int maxThreads = maxThreadsPerSM * numSMs;
    std::cout << "max supported #ele: " << maxThreads << std::endl;
    return maxThreads;
}

void Context::PollEvents() {
    auto& attrs = guiData->softBodyAttr;
    if (attrs.currSoftBodyId == -1) return;
    bool result = attrs.stiffness_0.second || attrs.stiffness_1.second || attrs.damp.second || attrs.muN.second || attrs.muT.second || attrs.getJumpDirty();
    if (result)
        mcrpSimContext->UpdateSingleSBAttr(guiData->softBodyAttr.currSoftBodyId, guiData->softBodyAttr);
    else
        return;
    attrs.stiffness_0.second = false;
    attrs.stiffness_1.second = false;
    attrs.damp.second = false;
    attrs.muN.second = false;
    attrs.muT.second = false;
}

void Context::LoadShaders(const std::string& vertShaderFilename, const std::string& fragShaderFilename)
{
    std::ifstream fileStream = utilityCore::findFile(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
    }
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();
    if (json.contains("shaders folder")) {
        std::string shadersFolder = json["shaders folder"];
        std::string vertShaderPath = shadersFolder + "/" + "lambert.vert.glsl";
        std::string fragShaderPath = shadersFolder + "/" + "lambert.frag.glsl";
        mpProgLambert->create(vertShaderPath.c_str(), fragShaderPath.c_str());
    }
    else {
        mpProgLambert->create(vertShaderFilename.c_str(), fragShaderFilename.c_str());
    }
    mpProgLambert->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    mpProgLambert->setCameraPos(cameraPosition);
    mpProgLambert->setModelMatrix(glm::mat4(1.f));
}

void Context::LoadFlatShaders(const std::string& vertShaderFilename, const std::string& fragShaderFilename)
{
    std::ifstream fileStream = utilityCore::findFile(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
    }
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();
    if (json.contains("shaders folder")) {
        std::string shadersFolder = json["shaders folder"];
        std::string vertShaderPath = shadersFolder + "/" + "flat.vert.glsl";
        std::string fragShaderPath = shadersFolder + "/" + "flat.frag.glsl";
        mpProgFlat->create(vertShaderPath.c_str(), fragShaderPath.c_str());
    }
    else {
        mpProgFlat->create(vertShaderFilename.c_str(), fragShaderFilename.c_str());
    }
    mpProgFlat->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    mpProgFlat->setCameraPos(cameraPosition);
    mpProgFlat->setModelMatrix(glm::mat4(1.f));
}

const std::vector<const char*>& Context::GetNamesSoftBodies() const {
    return mcrpSimContext->GetNamesSoftBodies();
}

int Context::GetNumQueries() const {
    return mcrpSimContext->GetNumQueries();
}

std::vector<FixedBody*> ReadFixedBodies(const nlohmann::json& json, const std::map<std::string, nlohmann::json>& fixedBodyDefs) {
    std::vector<FixedBody*>fixedBodies;
    for (const auto& fbJson : json) {
        auto& fbDefJson = fixedBodyDefs.at(std::string(fbJson["name"]));
        glm::vec3 pos;
        glm::vec3 scale;
        glm::vec3 rot;
        if (!fbJson.contains("pos")) {
            if (fbDefJson.contains("pos")) {
                pos = glm::vec3(fbDefJson["pos"][0].get<float>(), fbDefJson["pos"][1].get<float>(), fbDefJson["pos"][2].get<float>());
            }
            else {
                pos = glm::vec3(0.f);
            }
        }
        else {
            pos = glm::vec3(fbJson["pos"][0].get<float>(), fbJson["pos"][1].get<float>(), fbJson["pos"][2].get<float>());
        }
        if (!fbJson.contains("scale")) {
            if (fbDefJson.contains("scale")) {
                scale = glm::vec3(fbDefJson["scale"][0].get<float>(), fbDefJson["scale"][1].get<float>(), fbDefJson["scale"][2].get<float>());
            }
            else {
                scale = glm::vec3(1.f);
            }
        }
        else {
            scale = glm::vec3(fbJson["scale"][0].get<float>(), fbJson["scale"][1].get<float>(), fbJson["scale"][2].get<float>());
        }
        if (!fbJson.contains("rot")) {
            if (fbDefJson.contains("rot")) {
                rot = glm::vec3(fbDefJson["rot"][0].get<float>(), fbDefJson["rot"][1].get<float>(), fbDefJson["rot"][2].get<float>());
            }
            else {
                rot = glm::vec3(0.f);
            }
        }
        else {
            rot = glm::vec3(fbJson["rot"][0].get<float>(), fbJson["rot"][1].get<float>(), fbJson["rot"][2].get<float>());
        }
        if (fbDefJson["type"] == "sphere") {
            glm::mat4 model = utilityCore::modelMatrix(pos, rot, glm::vec3(1, 1, 1));
            int numSides;
            if (!fbJson.contains("numSides")) {
                if (fbDefJson.contains("numSides")) {
                    numSides = fbDefJson["numSides"].get<int>();
                }
                else {
                    numSides = 64;
                }
            }
            else {
                numSides = fbJson["numSides"].get<int>();
            }
            if (!fbJson.contains("radius")) {
                if (fbDefJson.contains("radius")) {
                    float radius = fbDefJson["radius"].get<float>();
                    fixedBodies.push_back(new Sphere(utilityCore::modelMatrix(pos, rot, glm::vec3(radius, radius, radius)), radius, numSides));
                }
                else {
                    fixedBodies.push_back(new Sphere(model, 1.f, numSides));
                }
            }
            else {
                float radius = fbJson["radius"].get<float>();
                fixedBodies.push_back(new Sphere(utilityCore::modelMatrix(pos, rot, glm::vec3(radius, radius, radius)), radius, numSides));
            }
        }
        if (fbDefJson["type"] == "cylinder") {
            glm::mat4 model = utilityCore::modelMatrix(pos, rot, glm::vec3(scale[0], scale[1], scale[0]));
            int numSides;
            if (!fbJson.contains("numSides")) {
                if (fbDefJson.contains("numSides")) {
                    numSides = fbDefJson["numSides"].get<int>();
                }
                else {
                    numSides = 64;
                }
            }
            else {
                numSides = fbJson["numSides"].get<int>();
            }
            fixedBodies.push_back(new Cylinder(model, scale, numSides));
        }
        if (fbDefJson["type"] == "plane") {
            fixedBodies.push_back(new Plane(utilityCore::modelMatrix(pos, rot, scale)));
        }
    }

    for (auto& fixedBody : fixedBodies) {
        fixedBody->create();
    }
    return fixedBodies;
}

SimulationCUDAContext* Context::LoadSimContext() {
    std::map<std::string, nlohmann::json>softBodyDefs;
    std::map<std::string, nlohmann::json>fixedBodyDefs;
    std::ifstream fileStream = utilityCore::findFile(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
        return nullptr;
    }
    int maxThreads = GetMaxCGThreads();
    SimulationCUDAContext::ExternalForce extForce;
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();
    int threadsPerBlock = 128, threadsPerBlockBVH = threadsPerBlock;
    if (json.contains("pause")) {
        pause = json["pause"].get<bool>();
    }
    int numIterations = 10;
    if (json.contains("num of iterations")) {
        numIterations = json["num of iterations"].get<int>();
    }
    if (json.contains("external force")) {
        auto& externalForceJson = json["external force"];
        if (externalForceJson.contains("jump")) {
            auto& jumpJson = externalForceJson["jump"];
            extForce.jump = glm::vec3(jumpJson[0].get<float>(), jumpJson[1].get<float>(), jumpJson[2].get<float>());
        }
    }
    if (json.contains("threads per block")) {
        threadsPerBlock = json["threads per block"].get<int>();
    }
    if (json.contains("threads per block(bvh)")) {
        threadsPerBlockBVH = json["threads per block(bvh)"].get<int>();
    }
    if (json.contains("softBodies")) {
        auto& softBodyJsons = json["softBodies"];
        for (auto& softBodyJson : softBodyJsons) {
            softBodyDefs[softBodyJson["name"]] = softBodyJson;
        }
    }
    if (json.contains("fixedBodies")) {
        auto& fixedBodyJsons = json["fixedBodies"];
        for (auto& fixedBodyJson : fixedBodyJsons) {
            fixedBodyDefs[fixedBodyJson["name"]] = fixedBodyJson;
        }
    }
    if (json.contains("contexts")) {
        auto& contextJsons = json["contexts"];
        for (auto& contextJson : contextJsons) {
            if (contextJson.contains("load") && !contextJson["load"].get<bool>())
                continue;
            std::string baseName = contextJson["name"];
            char* name = new char[baseName.size() + 1];
            strcpy(name, baseName.c_str());
            namesContexts.push_back(name);
            std::vector<FixedBody*> fixBodies;
            if (contextJson.contains("fixedBodies")) {
                fixBodies = ReadFixedBodies(contextJson["fixedBodies"], fixedBodyDefs);
            }
            mpSimContexts.push_back(new SimulationCUDAContext(this, baseName, extForce, contextJson, softBodyDefs, fixBodies, threadsPerBlock, threadsPerBlockBVH, maxThreads, numIterations));
            DOFs.push_back(mpSimContexts.back()->GetVertCnt() * 3);
            Eles.push_back(mpSimContexts.back()->GetTetCnt());
            spdlog::info("{} : #dof: {}, #ele: {}", baseName, DOFs.back(), Eles.back());
        }
        mcrpSimContext = mpSimContexts[0];
    }
    return mcrpSimContext;
}

void Context::InitDataContainer() {
    guiData->phi = phi;
    guiData->theta = theta;
    guiData->cameraLookAt = ogLookAt;
    guiData->zoom = zoom;
    guiData->Dt = mcrpSimContext->GetDt();
    guiData->Pause = false;
    guiData->UseEigen = mcrpSimContext->IsEigenGlobalSolver();
    guiData->softBodyAttr.currSoftBodyId = 0;
    guiData->currSimContextId = 0;
}

void Context::InitCuda() {
    LoadSimContext();

    // Clean up on program exit
    atexit(cleanupCuda);
}

void Context::Draw() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    mcrpSimContext->Draw(mpProgLambert, mpProgFlat);
}

void Context::Update() {
    PollEvents();
    if (panelModified) {
        if (guiData->currSimContextId != -1) {
            mcrpSimContext = mpSimContexts[guiData->currSimContextId];
        }
        mcrpSimContext->SetGlobalSolver(guiData->UseEigen);
        mcrpSimContext->SetCUDASolver(guiData->UseCUDASolver);
        mcrpSimContext->SetDt(guiData->Dt);
        phi = guiData->phi;
        theta = guiData->theta;
        mpCamera->lookAt = guiData->cameraLookAt;
        zoom = guiData->zoom;
        camchanged = true;
    }
    if (camchanged) {
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        mpCamera->view = -glm::normalize(cameraPosition);
        glm::vec3 v = mpCamera->view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(mpCamera->up);
        glm::vec3 r = glm::cross(v, u);
        mpCamera->up = glm::cross(r, v);
        mpCamera->right = r;

        mpCamera->position = cameraPosition;
        cameraPosition += mpCamera->lookAt;
        mpCamera->position = cameraPosition;
        guiData->phi = phi;
        guiData->theta = theta;
        guiData->zoom = zoom;
        camchanged = false;
        mpProgLambert->setCameraPos(cameraPosition);
        mpProgLambert->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
        mpProgFlat->setCameraPos(cameraPosition);
        mpProgFlat->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    }

    if (guiData->Reset) {
        iteration = 0;
        mcrpSimContext->Reset();
        mcrpSimContext->Update();
    }
    if (guiData->Pause) {
        pause = !pause;
    }
    if (!pause) {
        iteration++;
        mcrpSimContext->Update();
    }
    else if (guiData->Step) {
        iteration++;
        mcrpSimContext->Update();
    }
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    // execute the kernel
    // unmap buffer object
}

void Context::ResetCamera() {
    camchanged = true;
    mpCamera->lookAt = ogLookAt;
}


void cleanupCuda() {

}

void GuiDataContainer::SoftBodyAttr::setJump(bool val) {
    jump = { true, true };
}

void GuiDataContainer::SoftBodyAttr::setJumpClean(bool& val)
{
    if (jump.second) {
        val = jump.first;
        if (val) {
            jump = { false, true };
        }
        else {
            jump.second = false;
        }
    }
}

bool GuiDataContainer::SoftBodyAttr::getJumpDirty()const {
    return jump.second;
}

GuiDataContainer::GuiDataContainer()
    :mPQuery(new Query()), Dt(0.001), PointSize(15), LineWidth(10), WireFrame(false), BVHVis(false), BVHEnabled(true),
    handleCollision(true), QueryVis(false), QueryDebugMode(true), ObjectVis(true), Reset(false), Pause(false),
    Step(false), UseEigen(true), UseCUDASolver(true), CurrQueryId(0)
{
}

GuiDataContainer::~GuiDataContainer()
{
    delete mPQuery;
}
