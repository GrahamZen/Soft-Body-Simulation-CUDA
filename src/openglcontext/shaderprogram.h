#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <glm/glm.hpp>

#include <drawable.h>
#include <texture.h>


class ShaderProgram
{
public:

    GLuint vertShader; // A handle for the vertex shader stored in this shader program
    GLuint fragShader; // A handle for the fragment shader stored in this shader program
    GLuint prog;       // A handle for the linked shader program stored in this class

    int unifSampler2D; // A handle to the "uniform" sampler2D that will be used to read the texture containing the scene render
    int unifTime; // A handle for the "uniform" float representing time in the shader

public:
    ShaderProgram();
    virtual ~ShaderProgram();
    // Sets up the requisite GL data and shaders from the given .glsl files
    void create(const char* vertfile, const char* fragfile);
    // Sets up shader-specific handles
    virtual void setupMemberVars() = 0;
    // Tells our OpenGL context to use this shader to draw things
    void useMe();
    // Draw the given object to our screen using this ShaderProgram's shaders
    virtual void draw(Drawable& d, int textureSlot) = 0;
    virtual void drawPoints(Drawable& d) = 0;
    // Utility function used in create()
    char* textFileRead(const char*);
    // Utility function that prints any shader compilation errors to the console
    void printShaderInfoLog(int shader);
    // Utility function that prints any shader linking errors to the console
    void printLinkInfoLog(int prog);

    void setTime(int t);
};


#endif // SHADERPROGRAM_H
