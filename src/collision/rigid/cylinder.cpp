#include <cylinder.h>
#include <vector>
#include <glm/gtx/string_cast.hpp>

Cylinder::Cylinder()
{}

Cylinder::Cylinder(const glm::mat4& model, const glm::vec3& scale, int numSides) : FixedBody(model), m_radius(scale[0]), m_numSides(numSides), m_height(scale[1])
{}

std::ostream& operator<<(std::ostream& os, const Cylinder& Cylinder) {
    os << "Model: " << glm::to_string(Cylinder.m_model) << ", Radius: " << Cylinder.m_radius << ", NumSides: " << Cylinder.m_numSides << ", Height: " << Cylinder.m_height << std::endl;
    return os;
}

void Cylinder::create()
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec4> nor;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> idx;
    const float PI = 3.1415926f;
    float angleStep = 2 * PI / m_numSides;
    for (int i = 0; i <= m_numSides; ++i) {
        float angle = i * angleStep;
        float x = cos(angle);
        float z = sin(angle);
        pos.push_back(glm::vec3(x, 1.0, z));
        nor.push_back(glm::vec4(0, 1, 0, 0));
        pos.push_back(glm::vec3(x, -1.0, z));
        nor.push_back(glm::vec4(0, -1, 0, 0));
    }
    for (int i = 0; i <= m_numSides; ++i) {
        float angle = i * angleStep;
        float x = cos(angle);
        float z = sin(angle);
        pos.push_back(glm::vec3(x, 1.0, z));
        nor.push_back(glm::normalize(glm::vec4(x, 0, z, 0)));
        pos.push_back(glm::vec3(x, -1.0, z));
        nor.push_back(glm::normalize(glm::vec4(x, 0, z, 0)));
    }
    pos.push_back(glm::vec3(0, 1.0, 0));
    nor.push_back(glm::vec4(0, 1, 0, 0));
    pos.push_back(glm::vec3(0, -1.0, 0));
    nor.push_back(glm::vec4(0, -1, 0, 0));

    int topCenterIndex = pos.size() - 2;
    int bottomCenterIndex = pos.size() - 1;

    // Top cap
    for (int i = 0; i < m_numSides; ++i) {
        idx.push_back(topCenterIndex);
        idx.push_back((i + 1) * 2);
        idx.push_back(i * 2);
    }

    // Bottom cap
    for (int i = 0; i < m_numSides; ++i) {
        idx.push_back(bottomCenterIndex);
        idx.push_back(i * 2 + 1);
        idx.push_back((i + 1) * 2 + 1);
    }

    // Side
    for (int i = m_numSides + 1; i < m_numSides * 2 + 1; ++i) {
        idx.push_back(i * 2);
        idx.push_back(i * 2 + 1);
        idx.push_back((i + 1) * 2 + 1);

        idx.push_back(i * 2);
        idx.push_back((i + 1) * 2 + 1);
        idx.push_back((i + 1) * 2);
    }
    count = idx.size();


    generateIdx();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    generatePos();
    glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(glm::vec3), pos.data(), GL_STATIC_DRAW);

    generateNor();
    glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    glBufferData(GL_ARRAY_BUFFER, nor.size() * sizeof(glm::vec4), nor.data(), GL_STATIC_DRAW);
}

BodyType Cylinder::getType() const
{
    return BodyType::Cylinder;
}