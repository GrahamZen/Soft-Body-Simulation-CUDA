#pragma once

#include <GL/glew.h>
#include <memory>
#include <vector>

// Texture slot for the 2D HDR environment map
#define ENV_MAP_FLAT_TEX_SLOT 0
// Texture slot for the 3D HDR environment cube map
#define ENV_MAP_CUBE_TEX_SLOT 1
// Texture slot for the 3D HDR diffuse irradiance map
#define DIFFUSE_IRRADIANCE_CUBE_TEX_SLOT 2
// Texture slot for the 3D HDR glossy irradiance map
#define GLOSSY_IRRADIANCE_CUBE_TEX_SLOT 3
// Texture slot for the BRDF lookup texture
#define BRDF_LUT_TEX_SLOT 4

#define ALBEDO_TEX_SLOT 5
#define METALLIC_TEX_SLOT 6
#define ROUGHNESS_TEX_SLOT 7
#define AO_TEX_SLOT 8
#define NORMALS_TEX_SLOT 9
#define DISPLACEMENT_TEX_SLOT 10

void printGLErrorLog();

class Texture {
public:
    struct Image {
        int width, height;
        int channels;
        std::vector<unsigned char>data;
        Image(int width, int height, int channels, unsigned char*);
    }; 
    Texture();
    virtual ~Texture();

    virtual void create(const char *texturePath, bool wrap) = 0;
    void destroy();
    virtual void bind(GLuint texSlot);

    bool m_isCreated;

protected:
    GLuint m_textureHandle;
};

class Texture2D : public Texture {
public:
    Texture2D();
    ~Texture2D();

    void create(const char *texturePath, bool wrap) override;
    void load(GLuint texSlot);
private:
    std::shared_ptr<Image> m_textureImage;
};

class Texture2DHDR : public Texture {
public:
    Texture2DHDR();
    ~Texture2DHDR();

    void create(const char *texturePath, bool wrap) override;
};

class TextureCubemap : public Texture {
public:
    TextureCubemap();
    ~TextureCubemap();

    void create(const char *texturePath, bool wrap) override;
    void bind(GLuint texSlot) override;
};
