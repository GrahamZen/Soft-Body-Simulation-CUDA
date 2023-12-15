#pragma once

#include <rigid.h>
#include <glm/glm.hpp>


class Cylinder : public FixedBody
{
    friend std::ostream& operator<<(std::ostream& os, const Cylinder& Cylinder);
public:
    Cylinder();
    Cylinder(const glm::mat4& model, const glm::vec3& scale, int numSides = 16);

    virtual void create() override;
    virtual BodyType getType() const override;

    const float m_radius = 1.0f;
    const float m_height = 1.0f;
private:
    int m_numSides;
};

