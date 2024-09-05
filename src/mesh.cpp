#include <mesh.h>
#include <fstream>
#include <sstream>
#include <cuda_gl_interop.h>

Mesh::Mesh()
    : Drawable(),
    mp_texture(nullptr), mp_bgTexture(nullptr)
{
}

Mesh::~Mesh()
{
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(bufPos);
    cudaGLUnregisterBufferObject(bufNor);
}

void Mesh::createTetrahedron()
{
    // TODO: Create VBO data for positions, normals, UVs, and indices
    count = numTets * 12; // TODO: Set "count" to the number of indices in your index VBO
    std::vector<GLuint> triangles(count);
    for (int t = 0; t < numTets * 4; t++)
    {
        triangles[t * 3 + 0] = t * 3 + 0;
        triangles[t * 3 + 1] = t * 3 + 1;
        triangles[t * 3 + 2] = t * 3 + 2;
    }
    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(GLuint), triangles.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, numTets * 12 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    generateNor();
    glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    glBufferData(GL_ARRAY_BUFFER, numTets * 12 * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufNor_resource, bufNor, cudaGraphicsMapFlagsWriteDiscard);
}

void Mesh::createMesh()
{
    if (numTris == 0)return;
    // TODO: Create VBO data for positions, normals, UVs, and indices
    count = numTris * 3; // TODO: Set "count" to the number of indices in your index VBO
    std::vector<GLuint> triangles(count);
    for (int t = 0; t < numTris; t++)
    {
        triangles[t * 3 + 0] = t * 3 + 0;
        triangles[t * 3 + 1] = t * 3 + 1;
        triangles[t * 3 + 2] = t * 3 + 2;
    }
    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(GLuint), triangles.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, numTris * 9 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    generateNor();
    glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    glBufferData(GL_ARRAY_BUFFER, numTris * 9 * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufNor_resource, bufNor, cudaGraphicsMapFlagsWriteDiscard);
}

void Mesh::MapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);

    cudaGraphicsMapResources(1, &cuda_bufNor_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufNorDevPtr, &size, cuda_bufNor_resource);
}

void Mesh::UnMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_bufNor_resource, 0);
}

void Mesh::create()
{
    // Does nothing, as we have two separate VBO data
    // creation functions: createFromOBJ, which creates
    // our mesh VBOs from OBJ file data, and createCube,
    // which you will implement.
}

void Mesh::bindTexture() const
{
    mp_texture->bind(0);
}

void Mesh::loadTexture() const
{
    mp_texture->load(0);
}

void Mesh::bindBGTexture() const
{
    mp_bgTexture->bind(2);
}

void Mesh::loadBGTexture() const
{
    mp_bgTexture->load(2);
}
