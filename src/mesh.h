#pragma once

#include <drawable.h>
#include <texture.h>
#include <glm/glm.hpp>

class cudaGraphicsResource;

class Mesh : public Drawable
{
    friend class SoftBody;
public:
    Mesh();
    ~Mesh();
    void create() override;

    void createTetrahedron();
    void createCube();
    void createMesh();
    void bindTexture() const;
    void loadTexture() const;
    void bindBGTexture() const;
    void loadBGTexture() const;
    void MapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr);
    void UnMapDevicePtr();
protected:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
    cudaGraphicsResource* cuda_bufNor_resource = nullptr;
    int numTets = 0;
    int numTris = 0;
private:
    std::unique_ptr<Texture2D> mp_texture;
    std::unique_ptr<Texture2D> mp_bgTexture;
    const std::vector<glm::vec4> cube_normals{
    glm::vec4(0, 0, 1, 0),  // 0
    glm::vec4(0, -1, 0, 0), // 1
    glm::vec4(-1, 0, 0, 0), // 2
    glm::vec4(0, 0, 1, 0),  // 3
    glm::vec4(0, -1, 0, 0), // 4
    glm::vec4(1, 0, 0, 0),  // 5
    glm::vec4(0, 0, 1, 0),  // 6
    glm::vec4(0, 1, 0, 0),  // 7
    glm::vec4(1, 0, 0, 0),  // 8
    glm::vec4(0, 0, 1, 0),  // 9
    glm::vec4(0, 1, 0, 0),  // 10
    glm::vec4(-1, 0, 0, 0), // 11
    glm::vec4(0, 0, -1, 0), // 12
    glm::vec4(0, -1, 0, 0), // 13
    glm::vec4(-1, 0, 0, 0), // 14
    glm::vec4(0, 0, -1, 0), // 15
    glm::vec4(0, -1, 0, 0), // 16
    glm::vec4(1, 0, 0, 0),  // 17
    glm::vec4(0, 0, -1, 0), // 18
    glm::vec4(0, 1, 0, 0),  // 19
    glm::vec4(1, 0, 0, 0),  // 20
    glm::vec4(0, 0, -1, 0), // 21
    glm::vec4(0, 1, 0, 0),  // 22
    glm::vec4(-1, 0, 0, 0), // 23
    };
};