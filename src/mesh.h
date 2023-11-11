#pragma once

#include "drawable.h"
#include "texture.h"
#include <memory>
#include <string>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
class Mesh : public Drawable
{
    friend class SoftBody;
public:
    Mesh();
    ~Mesh();
    void create() override;

    void createFromOBJ(const char* filename, const char* textureFile, const char* bgTextureFile);
    void createCube(const char* textureFile = nullptr, const char* bgTextureFile = nullptr);
    void createTetrahedron();
    void bindTexture() const;
    void loadTexture() const;
    void bindBGTexture() const;
    void loadBGTexture() const;
protected:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
    cudaGraphicsResource* cuda_bufNor_resource = nullptr;
    int tet_number = 0;
private:
    std::unique_ptr<Texture> mp_texture;
    std::unique_ptr<Texture> mp_bgTexture;
};