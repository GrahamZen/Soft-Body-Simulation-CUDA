#pragma once

class GuiDataContainer;

std::string currentTimeString();
bool initOpenGL();
void mainLoop();
void cleanupOpenGL();
bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);