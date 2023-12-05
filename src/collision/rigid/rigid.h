#pragma once

#include <drawable.h>
#include <texture.h>
#include <string>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

class FixedBody : public Drawable
{
public:
    FixedBody(const glm::mat4& model);
    ~FixedBody() = default;
    virtual void create() = 0; // To be implemented by subclasses. Populates the VBOs of the Drawable.
    virtual float SDF(const glm::vec3& samplePoint) = 0;

protected:
    glm::mat4 m_model;
    glm::mat4 m_inverseModel;
};