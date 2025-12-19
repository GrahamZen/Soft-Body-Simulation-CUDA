#include <shaderprogram.h>
#include <fstream>
#include <sstream>
#include <filesystem> 
#include <iostream>
#include <cstring>

namespace fs = std::filesystem;

std::string readShaderSource(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.good()) {
        fs::path absolutePath = fs::absolute(filePath);
        std::cerr << "Cannot open shader file: " << absolutePath << std::endl;
        return "";
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

ShaderProgram::ShaderProgram()
    : vertShader(), fragShader(), prog(), unifSampler2D(-1), unifTime(-1)
{}

ShaderProgram::~ShaderProgram()
{}

void ShaderProgram::create(const char* vertfile, const char* fragfile)
{
    std::cout << "Setting up shader from " << vertfile << " and " << fragfile << std::endl;
    // Allocate space on our GPU for a vertex shader and a fragment shader and a shader program to manage the two
    vertShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    prog = glCreateProgram();
    // Get the body of text stored in our two .glsl files
    std::string lambertVS = readShaderSource(vertfile);
    std::string lambertFS = readShaderSource(fragfile);

    char* vertSource = new char[lambertVS.size() + 1];
    strcpy(vertSource, lambertVS.c_str());
    char* fragSource = new char[lambertFS.size() + 1];
    strcpy(fragSource, lambertFS.c_str());


    // Send the shader text to OpenGL and store it in the shaders specified by the handles vertShader and fragShader
    glShaderSource(vertShader, 1, &vertSource, 0);
    glShaderSource(fragShader, 1, &fragSource, 0);
    // Tell OpenGL to compile the shader text stored above
    glCompileShader(vertShader);
    glCompileShader(fragShader);
    // Check if everything compiled OK
    GLint compiled;
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        printShaderInfoLog(vertShader);
    }
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        printShaderInfoLog(fragShader);
    }

    // Tell prog that it manages these particular vertex and fragment shaders
    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);
    glLinkProgram(prog);

    // Check for linking success
    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        printLinkInfoLog(prog);
    }

    setupMemberVars();
}

void ShaderProgram::useMe()
{
    glUseProgram(prog);
}

void ShaderProgram::setTime(int t)
{
    useMe();

    if (unifTime != -1)
    {
        glUniform1i(unifTime, t);
    }
}

char* ShaderProgram::textFileRead(const char* fileName) {
    char* text;

    if (fileName != nullptr) {
        FILE* file = fopen(fileName, "rt");

        if (file != nullptr) {
            fseek(file, 0, SEEK_END);
            int count = ftell(file);
            rewind(file);

            if (count > 0) {
                text = (char*)malloc(sizeof(char) * (count + 1));
                count = fread(text, sizeof(char), count, file);
                text[count] = '\0';	//cap off the string with a terminal symbol, fixed by Cory
            }
            fclose(file);
        }
    }
    return text;
}

void ShaderProgram::printShaderInfoLog(int shader)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    // should additionally check for OpenGL errors here

    if (infoLogLen > 0)
    {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        std::cerr << "ShaderInfoLog:" << "/n" << infoLog << "/n";
        delete[] infoLog;
    }

    // should additionally check for OpenGL errors here
}

void ShaderProgram::printLinkInfoLog(int prog)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    // should additionally check for OpenGL errors here

    if (infoLogLen > 0) {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
        std::cerr << "LinkInfoLog:" << "/n" << infoLog << "/n";
        delete[] infoLog;
    }
}
