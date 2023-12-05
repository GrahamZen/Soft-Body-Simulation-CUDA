#pragma once

#include <glm/glm.hpp>
#include <rigid.h>
#include <iostream>

class Plane : public FixedBody
{
    friend std::ostream& operator<<(std::ostream& os, const Plane& Plane);
public:
    Plane();
    Plane(const glm::mat4& model, const glm::vec3& floorUp, float length);

    virtual void create() override;
    virtual BodyType getType() const override;

    glm::vec3 m_floorUp;
    float m_length;
};

