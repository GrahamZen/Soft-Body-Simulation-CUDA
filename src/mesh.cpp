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
void Mesh::createCube()
{
    // Code that sets up texture data on the GPU
    std::vector<glm::vec3> pos{
        glm::vec3(-1, -1, 1),  // 0
        glm::vec3(-1, -1, 1),  // 1
        glm::vec3(-1, -1, 1),  // 2
        glm::vec3(1, -1, 1),   // 3
        glm::vec3(1, -1, 1),   // 4
        glm::vec3(1, -1, 1),   // 5
        glm::vec3(1, 1, 1),    // 6
        glm::vec3(1, 1, 1),    // 7
        glm::vec3(1, 1, 1),    // 8
        glm::vec3(-1, 1, 1),   // 9
        glm::vec3(-1, 1, 1),   // 10
        glm::vec3(-1, 1, 1),   // 11
        glm::vec3(-1, -1, -1), // 12
        glm::vec3(-1, -1, -1), // 13
        glm::vec3(-1, -1, -1), // 14
        glm::vec3(1, -1, -1),  // 15
        glm::vec3(1, -1, -1),  // 16
        glm::vec3(1, -1, -1),  // 17
        glm::vec3(1, 1, -1),   // 18
        glm::vec3(1, 1, -1),   // 19
        glm::vec3(1, 1, -1),   // 20
        glm::vec3(-1, 1, -1),  // 21
        glm::vec3(-1, 1, -1),  // 22
        glm::vec3(-1, 1, -1) }; // 23
    // each uvs corresponds to a pos vec3, which corresponds to a normal.
    std::vector<glm::vec2> uvs{
        glm::vec2(0, 0),// 0
        glm::vec2(0, 1),// 1
        glm::vec2(1, 0),// 2
        glm::vec2(1, 0),// 3
        glm::vec2(1, 1),// 4
        glm::vec2(0, 0),// 5
        glm::vec2(1, 1),// 6
        glm::vec2(1, 0),// 7
        glm::vec2(0, 1),// 8
        glm::vec2(0, 1),// 9
        glm::vec2(0, 0),// 10
        glm::vec2(1, 1),// 11
        glm::vec2(1, 0),// 12
        glm::vec2(0, 0),// 13
        glm::vec2(0, 0),// 14
        glm::vec2(0, 0),// 15
        glm::vec2(1, 0),// 16
        glm::vec2(1, 0),// 17
        glm::vec2(0, 1),// 18
        glm::vec2(1, 1),// 19
        glm::vec2(1, 1),// 20
        glm::vec2(1, 1),// 21
        glm::vec2(0, 1),// 22
        glm::vec2(0, 1),// 23
    };

    std::vector<GLuint> idx{ 0, 3, 6,
                            0, 6, 9,
                            5, 17, 8,
                            8, 17, 20,
                            10, 7, 19,
                            10, 19, 22,
                            2, 14, 23,
                            2, 23, 11,
                            12, 15, 18,
                            12, 18, 21,
                            1, 4, 16,
                            1, 16, 13 };

    count = 36; // TODO: Set "count" to the number of indices in your index VBO

    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(glm::vec3), pos.data(), GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    generateNor();
    glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    glBufferData(GL_ARRAY_BUFFER, cube_normals.size() * sizeof(glm::vec4), cube_normals.data(), GL_DYNAMIC_DRAW);

    generateUV();
    glBindBuffer(GL_ARRAY_BUFFER, bufUV);
    glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), uvs.data(), GL_STATIC_DRAW);
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
