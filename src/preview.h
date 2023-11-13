#pragma once

extern GLuint vbo;

std::string currentTimeString();
bool initOpenGL();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);