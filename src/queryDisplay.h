#pragma once

#include <drawable.h>
#include <texture.h>
#include <string>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

class QueryDisplay : public Drawable
{
public:
    QueryDisplay();
    ~QueryDisplay();
    void create() override;
    virtual GLenum drawMode() override;

    void createQueries(int numVerts);
    void mapDevicePosPtr(glm::vec3** bufPosDevPtr, glm::vec4** bufCol);
    void unMapDevicePtr();
protected:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
    cudaGraphicsResource* cuda_bufCol_resource = nullptr;
    int numVerts;
};