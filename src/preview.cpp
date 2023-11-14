//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include <main.h>
#include <preview.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

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

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool initOpenGL() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(context->width, context->height, "CIS 565 Path Tracer", NULL, NULL);
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
    context->panelModified = false;
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Simulator Analytics");                  // Create a window called "Hello, world!" and append into it.
    ImGui::Checkbox("Wireframe mode", &imguiData->WireFrame);
    imguiData->Reset = ImGui::Button("Reset");
    bool dtChanged = ImGui::DragFloat("dt", &imguiData->Dt, 0.0001f, 0.0001f, 0.05f, "%.4f");
    float availWidth = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(availWidth * 0.25f);
    bool cameraPhiChanged = ImGui::DragFloat("Camera Phi", &imguiData->phi, 0.1f, -PI, PI, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(availWidth * 0.25f);
    bool cameraThetaChanged = ImGui::DragFloat("Camera Theta", &imguiData->theta, 0.1f, 0.001f, PI - 0.001f, "%.4f");
    bool cameraLookAtChanged = ImGui::DragFloat3("Camera Look At", &imguiData->cameraLookAt.x, 1.0f, -200.0f, 200.0f, "%.4f");
    bool zoomChanged = ImGui::DragFloat("Zoom", &imguiData->zoom, 10.f, 0.01f, 10000.0f, "%.4f");
    ImGui::Text("Soft Body Attributes");
    ImGui::Separator();
    imguiData->softBodyAttr.stiffness_0.second = ImGui::DragFloat("Stiffness 0", &imguiData->softBodyAttr.stiffness_0.first, 100.f, 0.0f, 100000.0f, "%.2f");
    imguiData->softBodyAttr.stiffness_1.second = ImGui::DragFloat("Stiffness 1", &imguiData->softBodyAttr.stiffness_1.first, 100.f, 0.0f, 100000.0f, "%.2f");
    imguiData->softBodyAttr.damp.second = ImGui::DragFloat("Damp", &imguiData->softBodyAttr.damp.first, 0.01f, 0.0f, 1.0f, "%.4f");
    imguiData->softBodyAttr.muN.second = ImGui::DragFloat("muN", &imguiData->softBodyAttr.muN.first, 0.01f, 0.0f, 100.0f, "%.4f");
    imguiData->softBodyAttr.muT.second = ImGui::DragFloat("muT", &imguiData->softBodyAttr.muT.first, 0.01f, 0.0f, 100.0f, "%.4f");
    ImGui::Separator();
    const auto& nameItems = context->GetnamesSoftBodies();
    if (ImGui::Combo("label", &imguiData->softBodyAttr.currSoftBodyId, nameItems.data(), nameItems.size()))
    {
    }

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
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    if (cameraPhiChanged || cameraThetaChanged || cameraLookAtChanged || zoomChanged || dtChanged) {
        context->panelModified = true;
    }

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
        context->Update();

        string title = "CIS565 SoftBody Simulation | " + utilityCore::convertIntToString(context->GetIteration()) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        if (imguiData->WireFrame)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // VAO, shader program, and texture already bound
        context->Draw();
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
