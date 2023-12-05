#include <glm/glm.hpp>
#include <rigid.h>

class Sphere : public FixedBody
{
public:
    Sphere(const glm::mat4& model, float radius, int numSides = 16);

    virtual void create() override;

    float SDF(const glm::vec3& samplePoint) override;

private:
    float m_radius;
    int m_numSides;
};

