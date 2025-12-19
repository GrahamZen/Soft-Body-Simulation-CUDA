#include <sceneStructs.h>
#include <main.h>
#include <preview.h>
#include <context.h>
#include <utilities.h>
#include <simulation/simulationContext.h>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cuda_runtime.h>
#include <iostream>

static std::string startTimeString;

// For camera controls
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

Context* context;

glm::mat4 Camera::getView()const
{
    return glm::lookAt(position, lookAt, up);
}
glm::mat4 Camera::getProj()const
{
    return glm::perspective(glm::radians(fov.y), resolution.x / (float)resolution.y, 0.1f, 1000.0f);
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();
    // Load scene file
    context = new Context("context.json");
    // Initialize CUDA and GL components
    initOpenGL();

    context->InitCuda();
    context->LoadShaders();
    context->LoadFlatShaders();
    // Initialize ImGui Data
    InitImguiData(context->guiData.get());
    context->InitDataContainer();
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error: " << err << std::endl;
    }
    // GLFW main loop
    mainLoop();
    cudaDeviceReset();

    delete context;
    return 0;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Camera& cam = *context->mpCamera;
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_R:
            context->ResetCamera();
            break;
        case GLFW_KEY_S:
            context->guiData->Step = true;
            break;
        case GLFW_KEY_SPACE:
            if (context->guiData->softBodyAttr.currSoftBodyId != -1)
                context->guiData->softBodyAttr.setJump(true);
            break;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        context->guiData->Step = true;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (MouseOverImGuiWindow())
    {
        return;
    }
    auto& mouseState = context->mouseState;
    mouseState.leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    mouseState.rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    mouseState.middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
    if (!mouseState.leftMousePressed && context->mcrpSimContext->GetMouseSelection().dragging)
    {
        context->mcrpSimContext->SetDragging(false);
        context->mcrpSimContext->ResetMoreDBC(true);
    }
    else
        context->mcrpSimContext->SetDragging(true);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    auto& mouseState = context->mouseState;
    if (context->mcrpSimContext->GetMouseSelection().dragging)
    {
        if (context->UpdateCursorPos(xpos, ypos, true))
            context->mcrpSimContext->ResetMoreDBC();
        else
            context->mcrpSimContext->SetDragging(false);
    }
    else
        context->mcrpSimContext->ResetMoreDBC(true);
    if (xpos == mouseState.lastPos.x || ypos == mouseState.lastPos.y)
        return; // otherwise, clicking back into window causes re-start
    double lastX = mouseState.lastPos.x;
    double lastY = mouseState.lastPos.y;
    if (mouseState.leftMousePressed) {
        if (!context->mcrpSimContext->GetMouseSelection().dragging) {
            // compute new camera parameters
            context->phi -= (xpos - lastX) / context->width * 3.f;
            context->theta -= (ypos - lastY) / context->height * 3.f;
            context->theta = std::fmax(0.001f, std::fmin(context->theta, PI));
            context->camchanged = true;
        }
    }
    if (mouseState.rightMousePressed) {
        context->zoom += (ypos - lastY) / context->height * 50.f;
        context->zoom = std::fmax(0.1f, context->zoom);
        context->camchanged = true;
    }
    if (mouseState.middleMousePressed) {
        Camera& cam = *context->mpCamera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.05f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.05f;
        context->camchanged = true;
    }
    mouseState.lastPos.x = xpos;
    mouseState.lastPos.y = ypos;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    context->mpCamera->resolution.y = height;
    context->mpCamera->resolution.x = width;
    context->mpCamera->computeCameraParams();
}
