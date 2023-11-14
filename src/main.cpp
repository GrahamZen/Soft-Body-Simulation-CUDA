#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <main.h>
#include <preview.h>
#include <cstring>
#include <surfaceshader.h>


static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

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
    // Initialize ImGui Data
    InitImguiData(context->guiData);
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
    if (context->guiData->softBodyAttr.jump.first == true)
        context->guiData->softBodyAttr.jump = { false, true };
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            context->ResetCamera();
            break;
        case GLFW_KEY_SPACE:
            if (context->guiData->softBodyAttr.currSoftBodyId != -1)
                context->guiData->softBodyAttr.jump = { true, true };
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (MouseOverImGuiWindow())
    {
        return;
    }
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
    if (leftMousePressed) {
        // compute new camera parameters
        context->phi -= (xpos - lastX) / context->width * 3.f;
        context->theta -= (ypos - lastY) / context->height * 3.f;
        context->theta = std::fmax(0.001f, std::fmin(context->theta, PI));
        context->camchanged = true;
    }
    else if (rightMousePressed) {
        context->zoom += (ypos - lastY) / context->height * 5.f;
        context->zoom = std::fmax(0.1f, context->zoom);
        context->camchanged = true;
    }
    else if (middleMousePressed) {
        Camera& cam = *context->mpCamera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        context->camchanged = true;
    }
    lastX = xpos;
    lastY = ypos;
}
