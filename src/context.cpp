#include <sceneStructs.h>
#include <surfaceshader.h>
#include <context.h>
#include <Mesh.h>
#include <collision/aabb.h>
#include <simulation/simulationContext.h>
#include <utilities.h>
#include <collision/rigid/sphere.h>
#include <collision/rigid/cylinder.h>
#include <collision/rigid/plane.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <iostream>
#include <fstream>
#include <sstream>

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

Context::Context(const std::string& _filename) :shaderType(ShaderType::PHONG), filename(_filename), mpCamera(new Camera(_filename)), mpProgLambert(new SurfaceShader()),
mpProgPhong(new SurfaceShader()), mpProgHighLight(new SurfaceShader()), mpProgFlat(new SurfaceShader()), mpProgSkybox(new SurfaceShader()),
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
}

Context::~Context()
{
    delete mpProgHighLight;
    delete mpProgLambert;
    delete mpProgPhong;
    delete mpProgFlat;
    delete mpProgSkybox;
    delete mcrpSimContext;
    delete guiData;
    delete mpCamera;
    delete mpCube;
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
    bool result = attrs.mu || attrs.lambda || attrs.damp || attrs.muN || attrs.muT || attrs.getJumpDirty();
    if (result)
        mcrpSimContext->UpdateSoftBodyAttr(guiData->softBodyAttr.currSoftBodyId, &guiData->softBodyAttr);
    else
        return;
    attrs.mu = false;
    attrs.lambda = false;
    attrs.damp = false;
    attrs.muN = false;
    attrs.muT = false;
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
        fragShaderPath = shadersFolder + "/" + "blinnphong.frag.glsl";
        mpProgPhong->create(vertShaderPath.c_str(), fragShaderPath.c_str());
        vertShaderPath = shadersFolder + "/" + "highLight.vert.glsl";
        fragShaderPath = shadersFolder + "/" + "highLight.frag.glsl";
        mpProgHighLight->create(vertShaderPath.c_str(), fragShaderPath.c_str());
    }
    else {
        mpProgLambert->create(vertShaderFilename.c_str(), fragShaderFilename.c_str());
    }
    mpProgLambert->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    mpProgLambert->setCameraPos(cameraPosition);
    mpProgLambert->setModelMatrix(glm::mat4(1.f));
    mpProgPhong->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    mpProgPhong->setCameraPos(cameraPosition);
    mpProgPhong->setModelMatrix(glm::mat4(1.f));
    mpProgHighLight->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    mpProgHighLight->setCameraPos(cameraPosition);
    mpProgHighLight->setModelMatrix(glm::mat4(1.f));
    if (json.contains("environmentMap")) {
        std::string envMapPath = json["environmentMap"]["path"];
        bool envMapOn = json["environmentMap"]["on"];
        if (envMapOn) {
            LoadEnvCubemap(envMapPath);
            mpProgSkybox->create("../src/shaders/envMap.vert.glsl", "../src/shaders/envMap.frag.glsl");
            mpProgSkybox->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
            mpProgSkybox->addUniform("u_EnvironmentMap");
            mpCube = new Mesh();
            mpCube->createCube();
        }
    }
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
        fixedBodies.back()->name = new char[std::string(fbJson["name"]).size() + 1];
        strcpy(fixedBodies.back()->name, std::string(fbJson["name"]).c_str());
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
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();
    int threadsPerBlock = 128, threadsPerBlockBVH = threadsPerBlock;
    if (json.contains("pause")) {
        pause = json["pause"].get<bool>();
    }
    if (json.contains("enable log")) {
        logEnabled = json["enable log"].get<bool>();
    }
    std::string filename = "logs/log_" + getCurrentTimeStamp() + ".txt";
    if (logEnabled) {
        auto logger = spdlog::basic_logger_mt("basic_logger", filename);
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::debug);
    }
    int numIterations = 10;
    if (json.contains("num of iterations")) {
        numIterations = json["num of iterations"].get<int>();
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
            std::vector<FixedBody*> fixBodies;
            if (contextJson.contains("fixedBodies")) {
                fixBodies = ReadFixedBodies(contextJson["fixedBodies"], fixedBodyDefs);
            }
            mpSimContexts.push_back(new SimulationCUDAContext(this, baseName, contextJson, softBodyDefs, fixBodies, threadsPerBlock, threadsPerBlockBVH, maxThreads, numIterations));
            DOFs.push_back(mpSimContexts.back()->GetVertCnt() * 3);
            Eles.push_back(mpSimContexts.back()->GetTetCnt());
            if (logEnabled)
                spdlog::info("{} #dof: {}, #ele: {}", "[" + baseName + "]", DOFs.back(), Eles.back());
        }
        mcrpSimContext = mpSimContexts[0];
    }
    return mcrpSimContext;
}

void Context::LoadEnvCubemap(const std::string& filename) {
    {
        envMap = new TextureCubemap();
        envMap->create(filename.c_str(), false);
    }
}

void Context::InitDataContainer() {
    guiData->phi = phi;
    guiData->theta = theta;
    guiData->cameraLookAt = ogLookAt;
    guiData->zoom = zoom;
    guiData->solverParams = mcrpSimContext->GetSolverParams();
    guiData->Pause = false;
    guiData->softBodyAttr.currSoftBodyId = -1;
    guiData->currSimContextId = 0;
}

void Context::InitCuda() {
    LoadSimContext();

    // Clean up on program exit
    atexit(cleanupCuda);
}

void Context::Draw() {
    if (mpProgSkybox && envMap->m_isCreated) {
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        envMap->bind(ENV_MAP_CUBE_TEX_SLOT);
        mpProgSkybox->setViewProjMatrix(glm::mat4(glm::mat3(mpCamera->getView())), mpCamera->getProj());
        mpProgSkybox->setUnifInt("u_EnvironmentMap", ENV_MAP_CUBE_TEX_SLOT);
        mpProgSkybox->draw(*mpCube);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    switch (shaderType)
    {
    case Context::ShaderType::LAMBERT:
        mcrpSimContext->Draw(mpProgHighLight, mpProgLambert, mpProgFlat, guiData->HighLightObjId);
        break;
    case Context::ShaderType::PHONG:
        mcrpSimContext->Draw(mpProgHighLight, mpProgPhong, mpProgFlat, guiData->HighLightObjId);
        break;
    default:
        break;
    }
}

void Context::SetBVHBuildType(int buildType)
{
    mcrpSimContext->SetBVHBuildType(buildType);
}

int& Context::GetBVHBuildType()
{
    return bvhBuildType;
}

void Context::SetShaderType(int shaderType)
{
    this->shaderType = (ShaderType)shaderType;
}

int& Context::GetShaderType()
{
    return (int&)shaderType;
}

void Context::Update() {
    PollEvents();
    if (panelModified) {
        if (guiData->currSimContextId != -1) {
            mcrpSimContext = mpSimContexts[guiData->currSimContextId];
            guiData->solverParams = mcrpSimContext->GetSolverParams();
        }
        mcrpSimContext->SetGlobalSolver(guiData->pdSolverType);
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
        switch (shaderType)
        {
        case Context::ShaderType::LAMBERT:
            mpProgLambert->setCameraPos(cameraPosition);
            mpProgLambert->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
            break;
        case Context::ShaderType::PHONG:
            mpProgPhong->setCameraPos(cameraPosition);
            mpProgPhong->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
            break;
        default:
            break;
        }
        mpProgHighLight->setCameraPos(cameraPosition);
        mpProgHighLight->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
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
        if (iteration == guiData->PauseIter)
            pause = true;
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

void SoftBodyAttr::setJump(bool val) {
    jump = { true, true };
}

void SoftBodyAttr::setJumpClean(bool& val)
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

bool SoftBodyAttr::getJumpDirty()const {
    return jump.second;
}

GuiDataContainer::GuiDataContainer()
    :mPQuery(new Query()), PointSize(15), LineWidth(10), WireFrame(false), BVHVis(false), BVHEnabled(true),
    handleCollision(true), QueryVis(false), QueryDebugMode(true), ObjectVis(true), Reset(false), Pause(false),
    Step(false), CurrQueryId(0)
{
}

GuiDataContainer::~GuiDataContainer()
{
    delete mPQuery;
}
