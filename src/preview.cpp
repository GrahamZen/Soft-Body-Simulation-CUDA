//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include <main.h>
#include <preview.h>
#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glfw.h>
#include <ImGui/imgui_impl_opengl3.h>
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void cleanupCuda() {

}

void initCuda() {
    simContext = new SimulationCUDAContext();
    simContext->softBodies.push_back(new SoftBody{ "../assets/house2.node", "../assets/house2.ele" });
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool initOpenGL() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    //initTextures();
    //glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.

    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //	counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        runCuda();

        string title = "CIS565 SoftBody Simulation | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // VAO, shader program, and texture already bound
        m_progLambert->draw(*(simContext->softBodies.front()));

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
