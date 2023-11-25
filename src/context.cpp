#include <context.h>
#include <sceneStructs.h>
#include <surfaceshader.h>
#include <simulationContext.h>
#include <iostream>
#include <fstream>

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
    std::ifstream fileStream = findFile(_filename);
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

Context::Context(const std::string& _filename) :filename(_filename), mpCamera(new Camera(_filename)), mpProgLambert(new SurfaceShader()),
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
    for (auto name : namesContexts) {
        delete[]name;
    }
    delete mpProgLambert;
    delete mcrpSimContext;
    delete guiData;
    delete mpCamera;
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
    mpProgLambert->create(vertShaderFilename.c_str(), fragShaderFilename.c_str());
    mpProgLambert->setViewProjMatrix(mpCamera->getView(), mpCamera->getProj());
    mpProgLambert->setCameraPos(cameraPosition);
    mpProgLambert->setModelMatrix(glm::mat4(1.f));
}
const std::vector<const char*>& Context::GetNamesSoftBodies() const {
    return mcrpSimContext->GetNamesSoftBodies();
}

SimulationCUDAContext* Context::LoadSimContext() {
    std::ifstream fileStream = findFile(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
        return nullptr;
    }
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();
    if (json.contains("contexts")) {
        auto& contextJsons = json["contexts"];
        for (auto& contextJson : contextJsons) {
            if (contextJson.contains("load") && !contextJson["load"].get<bool>())
                continue;
            std::string baseName = contextJson["name"];
            char* name = new char[baseName.size() + 1];
            strcpy(name, baseName.c_str());
            namesContexts.push_back(name);
            mpSimContexts.push_back(new SimulationCUDAContext(this, contextJson));
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

}

void Context::InitCuda() {
    LoadSimContext();
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void Context::Draw() {
    mcrpSimContext->Draw(mpProgLambert);
}

void Context::Update() {
    if (guiData->currSimContextId != -1) {
        mcrpSimContext = mpSimContexts[guiData->currSimContextId];
    }
    PollEvents();
    if (panelModified) {
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