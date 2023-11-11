#pragma once

extern GLuint vbo;

std::string currentTimeString();
bool initOpenGL();
void initCuda();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);