#include <rigid.h>

FixedBody::FixedBody() {}

FixedBody::FixedBody(const glm::mat4& model) : m_model(model), m_inverseModel(glm::inverse(model)) {}
