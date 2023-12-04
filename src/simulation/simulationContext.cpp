#include <simulationContext.h>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <filesystem> 
namespace fs = std::filesystem;

DataLoader::DataLoader(const int _threadsPerBlock) :threadsPerBlock(_threadsPerBlock)
{
}

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, const ExternalForce& _extForce, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, int _threadsPerBlock, int _threadsPerBlockBVH, int maxThreads)
    :context(ctx), extForce(_extForce), threadsPerBlock(_threadsPerBlock), m_bvh(_threadsPerBlockBVH, 1 << 25)
{
    DataLoader dataLoader(threadsPerBlock);
    if (json.contains("dt")) {
        dt = json["dt"].get<float>();
    }
    if (json.contains("gravity")) {
        gravity = json["gravity"].get<float>();
    }
    if (json.contains("pause")) {
        context->guiData->Pause = json["pause"].get<bool>();
    }
    if (json.contains("damp")) {
        float damp = json["damp"].get<float>();
    }
    if (json.contains("muN")) {
        float muN = json["muN"].get<float>();
    }
    if (json.contains("muT")) {
        float muT = json["muT"].get<float>();
    }
    if (json.contains("softBodies")) {
        for (const auto& sbJson : json["softBodies"]) {
            auto& sbDefJson = softBodyDefs.at(std::string(sbJson["name"]));
            std::string nodeFile = sbDefJson["nodeFile"];
            std::string eleFile = sbDefJson["eleFile"];
            std::string faceFile;
            if (sbDefJson.contains("faceFile")) {
                faceFile = sbDefJson["faceFile"];
            }
            glm::vec3 pos;
            glm::vec3 scale;
            glm::vec3 rot;
            float mass;
            float stiffness_0;
            float stiffness_1;
            int constraints;
            if (!sbJson.contains("pos")) {
                pos = glm::vec3(sbDefJson["pos"][0].get<float>(), sbDefJson["pos"][1].get<float>(), sbDefJson["pos"][2].get<float>());
            }
            else {
                pos = glm::vec3(sbJson["pos"][0].get<float>(), sbJson["pos"][1].get<float>(), sbJson["pos"][2].get<float>());
            }
            if (!sbJson.contains("scale")) {
                scale = glm::vec3(sbDefJson["scale"][0].get<float>(), sbDefJson["scale"][1].get<float>(), sbDefJson["scale"][2].get<float>());
            }
            else {
                scale = glm::vec3(sbJson["scale"][0].get<float>(), sbJson["scale"][1].get<float>(), sbJson["scale"][2].get<float>());
            }
            if (!sbJson.contains("rot")) {
                rot = glm::vec3(sbDefJson["rot"][0].get<float>(), sbDefJson["rot"][1].get<float>(), sbDefJson["rot"][2].get<float>());
            }
            else {
                rot = glm::vec3(sbJson["rot"][0].get<float>(), sbJson["rot"][1].get<float>(), sbJson["rot"][2].get<float>());
            }
            if (!sbJson.contains("mass")) {
                mass = sbDefJson["mass"].get<float>();
            }
            else {
                mass = sbJson["mass"].get<float>();
            }
            if (!sbJson.contains("stiffness_0")) {
                stiffness_0 = sbDefJson["stiffness_0"].get<float>();
            }
            else {
                stiffness_0 = sbJson["stiffness_0"].get<float>();
            }
            if (!sbJson.contains("stiffness_1")) {
                stiffness_1 = sbDefJson["stiffness_1"].get<float>();
            }
            else {
                stiffness_1 = sbJson["stiffness_1"].get<float>();
            }
            if (!sbJson.contains("constraints")) {
                constraints = sbDefJson["constraints"].get<int>();
            }
            else {
                constraints = sbJson["constraints"].get<int>();
            }
            bool centralize = sbDefJson["centralize"].get<bool>();
            int startIndex = sbDefJson["start index"].get<int>();
            std::string baseName = nodeFile.substr(nodeFile.find_last_of('/') + 1);
            char* name = new char[baseName.size() + 1];
            strcpy(name, baseName.c_str());
            namesSoftBodies.push_back(name);
            dataLoader.CollectData(nodeFile.c_str(), eleFile.c_str(), faceFile.c_str(), pos, scale, rot, centralize, startIndex,
                SoftBody::SoftBodyAttribute{ mass, stiffness_0, stiffness_1, constraints });
        }
    }
    dataLoader.AllocData(startIndices, dev_Xs, dev_X0s, dev_XTilts, dev_Vs, dev_Fs, dev_Edges, dev_Tets, dev_TetFathers, numVerts, numTets);
    for (auto softBodyData : dataLoader.m_softBodyData) {
        softBodies.push_back(new SoftBody(this, softBodyData.second, &softBodyData.first));
    }
    m_floor.createQuad(1000, floorY);
    m_bvh.Init(numTets, numVerts, maxThreads);
    cudaMalloc((void**)&dev_Normals, numVerts * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_tIs, numVerts * sizeof(dataType));
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
    shaderProgram->draw(m_floor, 0);
    if (context->guiData->ObjectVis) {
        for (auto softBody : softBodies)
            shaderProgram->draw(*softBody, 0);
    }
    if (context->guiData->handleCollision && context->guiData->BVHVis) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        shaderProgram->draw(m_bvh, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}

std::vector<GLuint> DataLoader::loadEleFile(const std::string& EleFilename, int startIndex, int& numTets)
{
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        fs::path absolutePath = fs::absolute(EleFilename);
        std::cerr << "Unable to open file: " << absolutePath << std::endl;
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

    file.close();
    return Tet;
}

std::vector<GLuint> DataLoader::loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris)
{
    std::string line;
    std::ifstream file(faceFilename);

    if (!file.is_open()) {
        // std::cerr << "Unable to open face file" << std::endl;
        return std::vector<GLuint>();
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numTris;

    std::vector<GLuint> Triangle(numTris * 3);

    int a, b, c, d, e;
    for (int tet = 0; tet < numTris && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Triangle[tet * 3 + 0] = b - startIndex;
        Triangle[tet * 3 + 1] = c - startIndex;
        Triangle[tet * 3 + 2] = d - startIndex;
    }

    file.close();
    return Triangle;
}

std::vector<glm::vec3> DataLoader::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
{
    std::ifstream file(nodeFilename);
    if (!file.is_open()) {
        fs::path absolutePath = fs::absolute(nodeFilename);
        std::cerr << "Unable to open file: " << absolutePath << std::endl;
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

    return X;
}

void DataLoader::CollectEdges(const std::vector<GLuint>& triIdx) {
    std::set<std::pair<GLuint, GLuint>> uniqueEdges;
    std::vector<GLuint> edges;

    for (size_t i = 0; i < triIdx.size(); i += 3) {
        GLuint v0 = triIdx[i];
        GLuint v1 = triIdx[i + 1];
        GLuint v2 = triIdx[i + 2];

        std::pair<GLuint, GLuint> edge1 = std::minmax(v0, v1);
        std::pair<GLuint, GLuint> edge2 = std::minmax(v1, v2);
        std::pair<GLuint, GLuint> edge3 = std::minmax(v2, v0);

        uniqueEdges.insert(edge1);
        uniqueEdges.insert(edge2);
        uniqueEdges.insert(edge3);
    }

    for (const auto& edge : uniqueEdges) {
        edges.push_back(edge.first);
        edges.push_back(edge.second);
    }

    m_edges.push_back(edges);
    totalNumEdges += edges.size() / 2;
}