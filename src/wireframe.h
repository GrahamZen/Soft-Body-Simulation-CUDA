#pragma once

#include <drawable.h>
#include <texture.h>
#include <string>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

class BVHNode;

class Wireframe : public Drawable
{
public:
    Wireframe();
    ~Wireframe();
    void create() override;
    virtual GLenum drawMode() override;

    void createBVH(int numNodes);
    void mapDevicePosPtr(glm::vec3** bufPosDevPtr);
    void unMapDevicePtr();
protected:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
};