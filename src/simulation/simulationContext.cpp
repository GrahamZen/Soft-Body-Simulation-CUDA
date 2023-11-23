#include <simulationContext.h>

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, nlohmann::json& json) :context(ctx)
{
    if (json.contains("dt")) {
        dt = json["dt"].get<float>();
    }
    if (json.contains("gravity")) {
        gravity = json["gravity"].get<float>();
    }

    if (json.contains("softBodies")) {
        for (const auto& sbJson : json["softBodies"]) {
            std::string nodeFile = sbJson["nodeFile"];
            std::string eleFile = sbJson["eleFile"];
            glm::vec3 pos = glm::vec3(sbJson["pos"][0].get<float>(), sbJson["pos"][1].get<float>(), sbJson["pos"][2].get<float>());
            glm::vec3 scale = glm::vec3(sbJson["scale"][0].get<float>(), sbJson["scale"][1].get<float>(), sbJson["scale"][2].get<float>());
            glm::vec3 rot = glm::vec3(sbJson["rot"][0].get<float>(), sbJson["rot"][1].get<float>(), sbJson["rot"][2].get<float>());
            bool jump = sbJson["jump"].get<bool>();
            float mass = sbJson["mass"].get<float>();
            float stiffness_0 = sbJson["stiffness_0"].get<float>();
            float stiffness_1 = sbJson["stiffness_1"].get<float>();
            float damp = sbJson["damp"].get<float>();
            float muN = sbJson["muN"].get<float>();
            float muT = sbJson["muT"].get<float>();
            int constraints = sbJson["constraints"].get<int>();
            bool centralize = sbJson["centralize"].get<bool>();
            int startIndex = sbJson["start index"].get<int>();
            std::string baseName = nodeFile.substr(nodeFile.find_last_of('/') + 1);
            char* name = new char[baseName.size() + 1];
            strcpy(name, baseName.c_str());
            ctx->namesSoftBodies.push_back(name);

            softBodies.push_back(new SoftBody(nodeFile.c_str(), eleFile.c_str(), this,
                pos, scale, rot,
                mass, stiffness_0, stiffness_1, damp, muN, muT, constraints,
                centralize, startIndex));
        }
    }
}

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto softbody : softBodies) {
        delete softbody;
    }
}
void SimulationCUDAContext::UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr) {
    softBodies[index]->setAttributes(softBodyAttr);
}

void SimulationCUDAContext::Reset()
{
    for (auto softbody : softBodies) {
        softbody->Reset();
    }
}

void SimulationCUDAContext::Draw(ShaderProgram* shaderProgram)
{
    for (auto softBody : softBodies)
        shaderProgram->draw(*softBody, 0);
}

AABB SimulationCUDAContext::GetAABB() const
{
    AABB result{ glm::vec3(FLT_MAX),glm::vec3(-FLT_MAX) };
    for (auto softBody : softBodies)
        result.expand(softBody->GetAABB());
    return result;
}

int SimulationCUDAContext::GetTetCnt() const
{
    int result = 0;
    for (auto softBody : softBodies)
        result += softBody->getTetNumber();
    return result;
}
