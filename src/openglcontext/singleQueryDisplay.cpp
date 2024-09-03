#include <singleQueryDisplay.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <vector>


SingleQueryDisplay::SingleQueryDisplay()
    : Drawable()
{
}

SingleQueryDisplay::~SingleQueryDisplay()
{
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(bufPos);
    cudaGLUnregisterBufferObject(bufVertPos);
    cudaGLUnregisterBufferObject(bufTriPos);
}

GLenum SingleQueryDisplay::drawMode()
{
    return GL_LINES;
}

void SingleQueryDisplay::create() {
    std::vector<GLuint> linesIdx{ 0, 1, 2, 3, 4, 5 };
    count = 6;
    std::vector<GLuint> triIdx{ 0, 1, 2 };

    std::vector<glm::vec4> colors(6, glm::vec4(1.0f, 0.0f, 0.0f, 0.6f));
    colors[4] = glm::vec4(0.0f, 1.0f, 1.0f, 0.8f);
    colors[5] = glm::vec4(0.0f, 1.0f, 1.0f, 0.8f);

    generateCol();
    glBindBuffer(GL_ARRAY_BUFFER, bufCol);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4), colors.data(), GL_STATIC_DRAW);

    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLuint), linesIdx.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufPos_resource, bufPos, cudaGraphicsMapFlagsWriteDiscard);

    // extra VBOs for vertex and triangle
    generateVertPos();
    glBindBuffer(GL_ARRAY_BUFFER, bufVertPos);
    glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufVertPos_resource, bufVertPos, cudaGraphicsMapFlagsWriteDiscard);

    generateTriPos();
    glBindBuffer(GL_ARRAY_BUFFER, bufTriPos);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_bufTriPos_resource, bufTriPos, cudaGraphicsMapFlagsWriteDiscard);

    generateTriIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufTriIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint), triIdx.data(), GL_STATIC_DRAW);
}

void SingleQueryDisplay::generateVertPos() {
    vertPosBound = true;
    glGenBuffers(1, &bufVertPos);
}

void SingleQueryDisplay::generateTriPos() {
    triPosBound = true;
    glGenBuffers(1, &bufTriPos);
}

void SingleQueryDisplay::generateTriIdx() {
    triIdxBound = true;
    glGenBuffers(1, &bufTriIdx);
}

bool SingleQueryDisplay::bindVertPos() {
    if (vertPosBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufVertPos);
    }
    return vertPosBound;
}

bool SingleQueryDisplay::bindTriPos() {
    if (triPosBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufTriPos);
    }
    return triPosBound;
}

bool SingleQueryDisplay::bindTriIdx() {
    if (triIdxBound) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufTriIdx);
    }
    return triIdxBound;
}

void SingleQueryDisplay::MapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec3** bufVertPosDevPtr, glm::vec3** bufTriPosDevPtr)
{
    size_t size;
    if (bufPosDevPtr) {
        cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);
    }

    if (bufVertPosDevPtr) {
        cudaGraphicsMapResources(1, &cuda_bufVertPos_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)bufVertPosDevPtr, &size, cuda_bufVertPos_resource);
    }

    if (bufTriPosDevPtr) {
        cudaGraphicsMapResources(1, &cuda_bufTriPos_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)bufTriPosDevPtr, &size, cuda_bufTriPos_resource);
    }
}

void SingleQueryDisplay::UnMapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec3** bufVertPosDevPtr, glm::vec3** bufTriPosDevPtr)
{
    if (bufPosDevPtr)
        cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    if (bufVertPosDevPtr)
        cudaGraphicsUnmapResources(1, &cuda_bufVertPos_resource, 0);
    if (bufTriPosDevPtr)
        cudaGraphicsUnmapResources(1, &cuda_bufTriPos_resource, 0);
}

void SingleQueryDisplay::SetCount(int cnt)
{
    count = cnt;
}
