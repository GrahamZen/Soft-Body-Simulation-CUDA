#pragma once

#include <drawable.h>
#include <glm/glm.hpp>

class Query;
class cudaGraphicsResource;

class SingleQueryDisplay : public Drawable
{
    // Drawable is responsible for drawing two lines
    // this class need separate VBOs for a vertex and a triangle
public:
    SingleQueryDisplay();
    ~SingleQueryDisplay();
    void create() override;
    virtual GLenum drawMode() override;

    void generateVertPos();
    void generateTriPos();
    void generateTriIdx();

    bool bindVertPos();
    bool bindTriPos();
    bool bindTriIdx();
    void MapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec3** bufVertPosDevPtr, glm::vec3** bufTriPosDevPtr);
    void UnMapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec3** bufVertPosDevPtr, glm::vec3** bufTriPosDevPtr);
    bool IsLine() const { return isLine; }
    void SetIsLine(bool isLine) { this->isLine = isLine; }
    void SetCount(int count);
private:
    cudaGraphicsResource* cuda_bufPos_resource = nullptr;
    cudaGraphicsResource* cuda_bufVertPos_resource = nullptr;
    cudaGraphicsResource* cuda_bufTriPos_resource = nullptr;
    GLuint bufVertPos = -1; // A Vertex Buffer Object that we will use to store mesh vertices (vec4s)
    GLuint bufTriPos = -1; // A Vertex Buffer Object that we will use to store mesh vertices (vec4s)
    GLuint bufTriIdx = -1; // A Vertex Buffer Object that we will use to store triangle indices (GLuints)

    bool vertPosBound = false;
    bool triIdxBound = false; // Set to TRUE by generateIdx(), returned by bindIdx().
    bool triPosBound = false;
    bool isLine = false;
};