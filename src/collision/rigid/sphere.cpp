#include <sphere.h>
#include <vector>
#include <glm/gtx/string_cast.hpp>

Sphere::Sphere()
{}

Sphere::Sphere(const glm::mat4& model, float radius, int numSides) : FixedBody(model), m_radius(radius), m_numSides(numSides)
{}

std::ostream& operator<<(std::ostream& os, const Sphere& sphere) {
    os << "Model: " << glm::to_string(sphere.m_model) << ", Radius: " << sphere.m_radius << ", NumSides: " << sphere.m_numSides;
    return os;
}

void Sphere::create()
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec4> nor;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> idx;

    const float PI = 3.1415926f;
    const float PI_2 = PI * 2.0f;
    const float PI_1_2 = PI * 0.5f;
    const float PI_1_4 = PI * 0.25f;

    for (int lat = 0; lat <= m_numSides; ++lat)
    {
        float theta = PI * float(lat) / float(m_numSides);
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int lon = 0; lon <= m_numSides; ++lon)
        {
            float phi = PI_2 * float(lon) / float(m_numSides);
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            glm::vec3 normal(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
            glm::vec3 vertex = normal * m_radius;
            pos.push_back(glm::vec3(m_model * glm::vec4(vertex, 1.0f)));
            nor.push_back(-glm::vec4(glm::normalize(glm::vec3(m_inverseTransposeModel * glm::vec4(normal, 0.0f))), 0.0f));
            uvs.push_back(glm::vec2(1.0f - float(lon) / float(m_numSides), 1.0f - float(lat) / float(m_numSides)));
        }
    }

    for (int i = 0; i < m_numSides; ++i)
    {
        for (int j = 0; j < m_numSides; ++j)
        {
            int i0 = i * (m_numSides + 1) + j;
            int i1 = i0 + 1;
            int i2 = i0 + (m_numSides + 1);
            int i3 = i2 + 1;

            idx.push_back(i0);
            idx.push_back(i2);
            idx.push_back(i1);

            idx.push_back(i1);
            idx.push_back(i2);
            idx.push_back(i3);
        }
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

BodyType Sphere::getType() const
{
    return BodyType::Sphere;
}