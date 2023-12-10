#include <iostream>
#include <fstream>
#include <sstream>
#include <wireframe.h>
#include <bvh.h>

Wireframe::Wireframe()
    : Drawable()
{
}

Wireframe::~Wireframe()
{
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(bufPos);
}

GLenum Wireframe::drawMode()
{
    return GL_LINES;
}

void Wireframe::createBVH(int numNodes) {
    const int verticesPerNode = 8;
    const std::vector<GLuint> lineIdx = {
        0, 1, 1, 2, 2, 3, 3, 0, // Front face
        4, 5, 5, 6, 6, 7, 7, 4, // Back face
        0, 4, 1, 5, 2, 6, 3, 7  // Connecting edges
    };
    const int indicesPerNode = lineIdx.size();

    std::vector<GLuint> idx;

    idx.reserve(numNodes * indicesPerNode);
    int startIndex = 0;
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < indicesPerNode; ++j) {
            idx.push_back(startIndex + lineIdx[j]);
        }
        startIndex += verticesPerNode;
    }

    count = idx.size();
    std::vector<glm::vec4> colors(count, glm::vec4(1.0f, 0.5f, 0.5f, 1.0f));

    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    generateCol();
    glBindBuffer(GL_ARRAY_BUFFER, bufCol);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4), colors.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, numNodes * verticesPerNode * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);
}

void Wireframe::MapDevicePosPtr(glm::vec3** bufPosDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);
}

void Wireframe::unMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
}

void Wireframe::create()
{
    // Does nothing, as we have two separate VBO data
    // creation functions: createFromOBJ, which creates
    // our mesh VBOs from OBJ file data, and createCube,
    // which you will implement.
}
