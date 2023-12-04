#include <drawable.h>

Drawable::Drawable()
    : bufIdx(), bufPos(), bufNor(), bufUV(),
    idxBound(false), posBound(false), norBound(false), uvBound(false)
{}


void Drawable::destroy()
{
    glDeleteBuffers(1, &bufIdx);
    glDeleteBuffers(1, &bufPos);
    glDeleteBuffers(1, &bufNor);
    glDeleteBuffers(1, &bufUV);
}

GLenum Drawable::drawMode()
{
    // Since we want every three indices in bufIdx to be
    // read to draw our Drawable, we tell that the draw mode
    // of this Drawable is GL_TRIANGLES

    // If we wanted to draw a wireframe, we would return GL_LINES

    return GL_TRIANGLES;
}

int Drawable::elemCount()
{
    return count;
}

void Drawable::generateIdx()
{
    idxBound = true;
    // Create a VBO on our GPU and store its handle in bufIdx
    glGenBuffers(1, &bufIdx);
}

void Drawable::generatePos()
{
    posBound = true;
    // Create a VBO on our GPU and store its handle in bufPos
    glGenBuffers(1, &bufPos);
}

void Drawable::generateNor()
{
    norBound = true;
    // Create a VBO on our GPU and store its handle in bufNor
    glGenBuffers(1, &bufNor);
}

void Drawable::generateUV()
{
    uvBound = true;
    // Create a VBO on our GPU and store its handle in bufCol
    glGenBuffers(1, &bufUV);
}

void Drawable::generateCol() {
    colBound = true;
    glGenBuffers(1, &bufCol);
}

bool Drawable::bindCol() {
    if (colBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufCol);
        return true;
    }
    return false;
}

bool Drawable::bindIdx()
{
    if (idxBound) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    }
    return idxBound;
}

bool Drawable::bindPos()
{
    if (posBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    }
    return posBound;
}

bool Drawable::bindNor()
{
    if (norBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    }
    return norBound;
}

bool Drawable::bindUV()
{
    if (uvBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufUV);
    }
    return uvBound;
}
