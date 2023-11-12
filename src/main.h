#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include <sceneStructs.h>
#include <simulationContext.h>
#include <surfaceshader.h>
#include <mesh.h>

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Camera* camera;
extern int iteration;

extern int width;
extern int height;

extern SurfaceShader* m_progLambert;
extern Mesh* m_mesh;
extern SimulationCUDAContext* simContext;

void runCuda();
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
