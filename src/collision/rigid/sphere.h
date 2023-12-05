#pragma once

#include <glm/glm.hpp>
#include <rigid.h>
#include <iostream>

class Sphere : public FixedBody
{
    friend std::ostream& operator<<(std::ostream& os, const Sphere& sphere);
public:
    Sphere();
    Sphere(const glm::mat4& model, float radius, int numSides = 16);

    virtual void create() override;
    virtual BodyType getType() const override;

    float m_radius;

private:
    int m_numSides;
};

