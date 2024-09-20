#include <queryDisplay.h>
#include <fstream>
#include <sstream>

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
    return GL_LINES;
}

void QueryDisplay::createQueries(int _numVerts) {
    numVerts = _numVerts;
    count = numVerts * 2;
    std::vector<GLuint> lines(count);
    std::vector<glm::vec4> color(count, glm::vec4(0.0f, 1.0f, 1.0f, 1.0f));
    for (int t = 0; t < numVerts; t++)
    {
        lines[t * 2 + 0] = t;
        lines[t * 2 + 1] = t + numVerts;
    }

    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(GLuint), lines.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    generateCol();
    glBindBuffer(GL_ARRAY_BUFFER, bufCol);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(glm::vec4), color.data(), GL_DYNAMIC_DRAW);
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
