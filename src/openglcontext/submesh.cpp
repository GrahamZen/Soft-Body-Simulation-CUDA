#include <submesh.h>
#include <fstream>
#include <sstream>
#include <cuda_gl_interop.h>

SubMesh::SubMesh()
    : Drawable()
{
}

SubMesh::~SubMesh()
{
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(bufPos);
    cudaGLUnregisterBufferObject(bufNor);
}

void SubMesh::createSubMesh()
{
    // TODO: Create VBO data for positions, normals, UVs, and indices
    count = numTris * 3; // TODO: Set "count" to the number of indices in your index VBO
    std::vector<GLuint> triangles(count);
    for (int t = 0; t < numTris; t++)
    {
        triangles[t * 3 + 0] = t * 3 + 0;
        triangles[t * 3 + 1] = t * 3 + 1;
        triangles[t * 3 + 2] = t * 3 + 2;
    }
    std::vector<glm::vec4> colors(count, glm::vec4(0, 1, 0, 1));
    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(GLuint), triangles.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    generateNor();
    glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufNor_resource, bufNor, cudaGraphicsMapFlagsWriteDiscard);

    generateCol();
    glBindBuffer(GL_ARRAY_BUFFER, bufCol);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(glm::vec4), colors.data(), GL_STATIC_DRAW);
}

void SubMesh::MapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);

    cudaGraphicsMapResources(1, &cuda_bufNor_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufNorDevPtr, &size, cuda_bufNor_resource);
}

void SubMesh::UnMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_bufNor_resource, 0);
}

void SubMesh::create()
{}