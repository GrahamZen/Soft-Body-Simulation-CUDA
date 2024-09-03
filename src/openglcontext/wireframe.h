#pragma once

#include <drawable.h>
#include <texture.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

template<typename Scalar>
class BVHNode;

class Wireframe : public Drawable
{
public:
    Wireframe();
    ~Wireframe();
    void create() override;
    virtual GLenum drawMode() override;

    void createBVH(int numNodes);
    void MapDevicePosPtr(glm::vec3** bufPosDevPtr);
    void UnMapDevicePtr();
protected:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
};