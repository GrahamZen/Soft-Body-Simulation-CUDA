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

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

GuiDataContainer* guiData;
Camera* camera;

SurfaceShader* m_progLambert;
SimulationCUDAContext* simContext;

int iteration;

int width = 1024;
int height = 1024;

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

    camera = new Camera{ glm::ivec2(width, height),
        glm::vec3(0, 0, 10), glm::vec3(0, 0, 0),
        glm::vec3(0, 0, -1), glm::vec3(0, 1, 0), glm::vec3(1, 0, 0),
        glm::vec2(45, 45), glm::vec2(0.001f, 0.001f) };
    Camera& cam = *camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();


    // Initialize CUDA and GL components
    initOpenGL();
    m_progLambert = new SurfaceShader();
    m_progLambert->create("../src/shaders/lambert.vert.glsl", "../src/shaders/lambert.frag.glsl");
    m_progLambert->setViewProjMatrix(cam.getView(), cam.getProj());
    m_progLambert->setCameraPos(cameraPosition);
    m_progLambert->setModelMatrix(glm::mat4(1.f));

    initCuda();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error: " << err << std::endl;
    }
    // GLFW main loop
    mainLoop();
    cudaDeviceReset();

    delete m_progLambert;
    delete simContext;
    return 0;
}

void runCuda() {
    if (camchanged) {
        Camera& cam = *camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
        m_progLambert->setCameraPos(cameraPosition);
        m_progLambert->setViewProjMatrix(camera->getView(), camera->getProj());
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    iteration++;
    // execute the kernel
    simContext->Update();
    // unmap buffer object

}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    simContext->softBodies.front()->setJump(false);
    Camera& cam = *camera;
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            camchanged = true;
            cam.lookAt = ogLookAt;
            break;
        case GLFW_KEY_SPACE:
            simContext->softBodies.front()->setJump(true);
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
        phi -= (xpos - lastX) / width * 3.f;
        theta -= (ypos - lastY) / height * 3.f;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed) {
        zoom += (ypos - lastY) / height * 5.f;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed) {
        Camera& cam = *camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }
    lastX = xpos;
    lastY = ypos;
}
