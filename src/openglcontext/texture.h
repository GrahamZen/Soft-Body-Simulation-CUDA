#pragma once

#include <GL/glew.h>
#include <memory>
#include <vector>

void printGLErrorLog();

class Texture
{
public:
    struct Image {
        int width, height;
        int channels;
        std::vector<unsigned char>data;
        Image(int width, int height, int channels, unsigned char*);
    };
    Texture();
    ~Texture();

    void create(const char* texturePath);
    void load(int texSlot);
    void bind(int texSlot);

private:
    GLuint m_textureHandle;
    std::shared_ptr<Image> m_textureImage;
};
