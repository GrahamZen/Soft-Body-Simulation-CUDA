#include <rigid.h>

FixedBody::FixedBody() {}

FixedBody::FixedBody(const glm::mat4& model) : m_model(model), m_inverseModel(glm::inverse(model)), m_inverseTransposeModel(glm::transpose(glm::inverse(model))) {}

FixedBody::~FixedBody()
{
    // an error occurs when name is overwritten
    if(name)
        delete []name;
}
