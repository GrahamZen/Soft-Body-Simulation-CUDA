#include <plane.h>
#include <glm/gtx/string_cast.hpp>
#include <vector>
#include <iostream>

Plane::Plane()
{}

Plane::Plane(const glm::mat4& model) : FixedBody(model), m_floorUp(glm::normalize(glm::vec3(m_inverseTransposeModel* glm::vec4(0, 1, 0, 0))))
{}

std::ostream& operator<<(std::ostream& os, const Plane& Plane) {
    os << "Model: " << glm::to_string(Plane.m_model) << ", FloorUp: " << glm::to_string(Plane.m_floorUp);
    return os;
}

void Plane::create()
{
    std::vector<glm::vec3> pos{
        glm::vec3(-1, 0, -1),
        glm::vec3(-1, 0, 1),
        glm::vec3(1, 0, -1),
        glm::vec3(1, 0, 1),
    };
    std::vector<glm::vec4> nor(pos.size(), glm::vec4(0, 1, 0, 0));
    // each uvs corresponds to a pos vec3, which corresponds to a normal.
    std::vector<glm::vec2> uvs{
        glm::vec2(0, 0),
        glm::vec2(0, 1),
        glm::vec2(1, 0),
        glm::vec2(1, 1)
    };

    std::vector<indexType> idx{ 0, 1, 2, 2, 1, 3 };

    count = 6; // TODO: Set "count" to the number of indices in your index VBO

    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(indexType), idx.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(glm::vec3), pos.data(), GL_STATIC_DRAW);

    generateNor();
    glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    glBufferData(GL_ARRAY_BUFFER, nor.size() * sizeof(glm::vec4), nor.data(), GL_STATIC_DRAW);

    generateUV();
    glBindBuffer(GL_ARRAY_BUFFER, bufUV);
    glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), uvs.data(), GL_STATIC_DRAW);
}

BodyType Plane::getType() const
{
    return BodyType::Plane;
}