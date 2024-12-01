#pragma once

class GuiDataContainer;

std::string currentTimeString();
bool initOpenGL();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);