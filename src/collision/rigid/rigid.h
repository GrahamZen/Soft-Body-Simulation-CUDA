#pragma once

#include <drawable.h>
#include <texture.h>
#include <string>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

enum class BodyType {
    Sphere,
    Plane,
    Cylinder
};

class FixedBody : public Drawable
{
public:
    FixedBody();
    FixedBody(const glm::mat4& model);
    ~FixedBody() = default;
    virtual BodyType getType() const = 0;
    virtual void create() = 0; // To be implemented by subclasses. Populates the VBOs of the Drawable.
    glm::mat4 m_model;
    glm::mat4 m_inverseModel;
    glm::mat4 m_inverseTransposeModel;
};