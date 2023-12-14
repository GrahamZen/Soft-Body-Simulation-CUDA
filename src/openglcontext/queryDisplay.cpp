#include <iostream>
#include <fstream>
#include <sstream>
#include <queryDisplay.h>
#include <collision/bvh.h>

QueryDisplay::QueryDisplay()
    : Drawable()
{
}

QueryDisplay::~QueryDisplay()
{
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(bufPos);
    cudaGLUnregisterBufferObject(bufCol);
}

GLenum QueryDisplay::drawMode()
{
    return GL_POINTS;
}

void QueryDisplay::createQueries(int _numVerts) {
    numVerts = _numVerts;
    count = numVerts;
    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    generateCol();
    glBindBuffer(GL_ARRAY_BUFFER, bufCol);
    glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufCol_resource, bufCol, cudaGraphicsMapFlagsWriteDiscard);
}

void QueryDisplay::MapDevicePosPtr(glm::vec3** bufPosDevPtr, glm::vec4** bufColDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);
    if (bufColDevPtr) {
        cudaGraphicsMapResources(1, &cuda_bufCol_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)bufColDevPtr, &size, cuda_bufCol_resource);
    }
}

void QueryDisplay::UnMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_bufCol_resource, 0);
}

void QueryDisplay::create()
{
    // Does nothing, as we have two separate VBO data
    // creation functions: createFromOBJ, which creates
    // our mesh VBOs from OBJ file data, and createCube,
    // which you will implement.
}
