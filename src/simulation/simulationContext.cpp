#include <simulationContext.h>
#include <fstream>
#include <sstream>

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, nlohmann::json& json) :context(ctx)
{
    DataLoader dataLoader;
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
            dataLoader.CollectData(nodeFile.c_str(), eleFile.c_str(), pos, scale, rot, centralize, startIndex,
                SoftBody::SoftBodyAttribute{ mass, stiffness_0, stiffness_1, damp, muN, muT, constraints });
        }
    }
    dataLoader.AllocData(startIndices, dev_Xs, dev_X0s, dev_Vs, dev_Fs, dev_Tets, numVerts, numTets);
    for (auto softBodyData : dataLoader.m_softBodyData) {
        softBodies.push_back(new SoftBody(this, softBodyData.second, &softBodyData.first));
    }
    m_bvh.Init(GetTetCnt(), softBodies.size(), GetVertCnt());
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

int SimulationCUDAContext::GetVertCnt() const
{
    int result = 0;
    for (auto softBody : softBodies)
        result += softBody->getNumber();
    return result;
}

void SimulationCUDAContext::CCD()
{
    //auto pairCollision = m_bvh.detectCollisionCandidates(Tet, numTets, X, numVerts);
}

std::vector<GLuint> DataLoader::loadEleFile(const std::string& EleFilename, int startIndex, int& numTets)
{
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numTets;

    std::vector<GLuint> Tet(numTets * 4);

    int a, b, c, d, e;
    for (int tet = 0; tet < numTets && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Tet[tet * 4 + 0] = b - startIndex;
        Tet[tet * 4 + 1] = c - startIndex;
        Tet[tet * 4 + 2] = d - startIndex;
        Tet[tet * 4 + 3] = e - startIndex;
    }
    std::cout << "number of tetrahedrons: " << numTets << std::endl;

    file.close();
    return Tet;
}
std::vector<glm::vec3> DataLoader::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
{
    std::ifstream file(nodeFilename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << nodeFilename << std::endl;
        return {};
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numVerts;
    std::vector<glm::vec3> X(numVerts);
    glm::vec3 center(0.0f);

    for (int i = 0; i < numVerts && std::getline(file, line); ++i) {
        std::istringstream lineStream(line);
        int index;
        float x, y, z;
        lineStream >> index >> x >> y >> z;

        X[i].x = x;
        X[i].y = y;
        X[i].z = z;

        center += X[i];
    }

    // Centralize the model
    if (centralize) {
        center /= static_cast<float>(numVerts);
        for (int i = 0; i < numVerts; ++i) {
            X[i] -= center;
            float temp = X[i].y;
            X[i].y = X[i].z;
            X[i].z = temp;
        }
    }
    std::cout << "number of nodes: " << numVerts << std::endl;

    return X;
}