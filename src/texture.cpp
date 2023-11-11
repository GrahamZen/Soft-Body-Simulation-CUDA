#include "texture.h"
#include "tinyobj/stb_image.h"
#include <iostream>
#include <string>

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

Texture::Texture()
    : m_textureHandle(-1), m_textureImage(nullptr)
{}

Texture::~Texture()
{}

void Texture::create(const char* texturePath)
{
    printGLErrorLog();

    int width, height, channels;
    unsigned char* data = stbi_load(texturePath, &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << texturePath << std::endl;
        return;
    }
    m_textureImage = std::make_shared<Image>(width, height, channels, data);
    glGenTextures(1, &m_textureHandle);

    printGLErrorLog();
}

void Texture::load(int texSlot = 0)
{
    printGLErrorLog();

    glActiveTexture(GL_TEXTURE0 + texSlot);
    glBindTexture(GL_TEXTURE_2D, m_textureHandle);

    // These parameters need to be set for EVERY texture you create
    // They don't always have to be set to the values given here, but they do need
    // to be set
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        m_textureImage->width, m_textureImage->height,
        0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, m_textureImage->data.data());
    printGLErrorLog();
}


void Texture::bind(int texSlot = 0)
{
    glActiveTexture(GL_TEXTURE0 + texSlot);
    glBindTexture(GL_TEXTURE_2D, m_textureHandle);
}

Texture::Image::Image(int width, int height, int channels, unsigned char* data)
    :width(width), height(height), channels(channels), data(std::vector<unsigned char>(data, data + width * height * channels))
{
}
