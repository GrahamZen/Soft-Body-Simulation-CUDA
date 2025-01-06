#pragma once

#include <drawable.h>
#include <texture.h>
#include <glm/glm.hpp>

class cudaGraphicsResource;

class SubMesh : public Drawable
{
    friend class SoftBody;
public:
    SubMesh();
    ~SubMesh();
    void create() override;
    void createSubMesh();
    void MapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr);
    void UnMapDevicePtr();
protected:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
    cudaGraphicsResource* cuda_bufNor_resource = nullptr;
    int numTris = 1;
};