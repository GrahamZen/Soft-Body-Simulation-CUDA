#include <texture.h>
#include "stb_image.h"
#include <iostream>
#include <string>
#include <array>

void printGLErrorLog()
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error " << error << ": ";
        const char* e =
            error == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION" :
            error == GL_INVALID_ENUM ? "GL_INVALID_ENUM" :
            error == GL_INVALID_VALUE ? "GL_INVALID_VALUE" :
            error == GL_INVALID_INDEX ? "GL_INVALID_INDEX" :
            error == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION" : std::to_string(error).c_str();
        std::cerr << e << std::endl;
        // Throwing here allows us to use the debugger to track down the error.
#ifndef __APPLE__
        // Don't do this on OS X.
        // http://lists.apple.com/archives/mac-opengl/2012/Jul/msg00038.html
        throw;
#endif
    }
}

void Texture::bind(GLuint texSlot = 0) {
    glActiveTexture(GL_TEXTURE0 + texSlot);
    glBindTexture(GL_TEXTURE_2D, m_textureHandle);
}

void TextureCubemap::bind(GLuint texSlot = 0) {
    glActiveTexture(GL_TEXTURE0 + texSlot);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureHandle);
}

Texture::Texture()
    : m_textureHandle(0), m_isCreated(false)
{}

Texture2D::Texture2D()
    : Texture(), m_textureImage(nullptr)
{}

Texture2DHDR::Texture2DHDR()
    : Texture()
{}

TextureCubemap::TextureCubemap()
    : Texture()
{}

Texture::~Texture()
{}
Texture2D::~Texture2D()
{}
TextureCubemap::~TextureCubemap()
{}
Texture2DHDR::~Texture2DHDR()
{}

void Texture::destroy() {
    glDeleteTextures(1, &m_textureHandle);
    m_isCreated = false;
}

void Texture2D::create(const char* texturePath, bool wrap) {
    printGLErrorLog();

    int width, height, channels;
    unsigned char* data = stbi_load(texturePath, &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << texturePath << std::endl;
        return;
    }
    Image img(width, height, channels, data);

    //    m_textureImage = mkU<Image>(img);
    glGenTextures(1, &m_textureHandle);

    glBindTexture(GL_TEXTURE_2D, m_textureHandle);

    // These parameters need to be set for EVERY texture you create
    // They don't always have to be set to the values given here, but they do need
    // to be set
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    if (wrap) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        img.width, img.height,
        0, GL_BGRA, GL_UNSIGNED_BYTE, img.data.data());
    printGLErrorLog();
    m_isCreated = true;
}

void Texture2D::load(GLuint texSlot)
{
    
}

void Texture2DHDR::create(const char* texturePath, bool wrap) {
    stbi_set_flip_vertically_on_load(true);
    int width, height, nrComponents;
    float* data = stbi_loadf(texturePath, &width, &height, &nrComponents, 0);
    if (data) {
        glGenTextures(1, &m_textureHandle);
        glBindTexture(GL_TEXTURE_2D, m_textureHandle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (wrap) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        }
        else {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        stbi_image_free(data);
    }
    else {
        throw std::runtime_error("Failed to load HDR image!");
    }
    m_isCreated = true;
}
//  "negx", "negy" ,"negz","posx","posy" ,"posz" 
constexpr std::array<const char*, 6> cubemapPostfix = { "posx", "negx", "posy", "negy", "posz", "negz" };

void TextureCubemap::create(const char* texturePath, bool wrap) {
    glGenTextures(1, &m_textureHandle);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureHandle);

    for (int i = 0; i < 6; i++) {
        std::string path = std::string(texturePath) + "/" + cubemapPostfix[i] + ".jpg";
        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!data) {
            std::cerr << "Failed to load texture: " << path << std::endl;
            return;
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB,
            width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    }
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    if (wrap) {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    else {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    m_isCreated = true;
}

Texture::Image::Image(int width, int height, int channels, unsigned char* data)
    :width(width), height(height), channels(channels), data(std::vector<unsigned char>(data, data + width * height * channels))
{
}
