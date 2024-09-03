#pragma once

#include <context.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Context* context;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);