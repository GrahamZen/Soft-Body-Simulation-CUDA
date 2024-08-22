#include <utilities.h>
#include <surfaceshader.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/simulationContext.h>
#include <simulation/softBody.h>
#include <spdlog/spdlog.h>
#include <map>
#include <chrono>
#include <functional>

template<typename Func>
void measureExecutionTime(const Func& func, const std::string& message, bool print = false) {
    if (!print) {
        func();
        return;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("{} Time: {} milliseconds", message, milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>& _fixedBodies, int _threadsPerBlock, int _threadsPerBlockBVH, int maxThreads, int _numIterations)
    :context(ctx), threadsPerBlock(_threadsPerBlock), fixedBodies(_fixedBodies), name(_name)
{
    DataLoader dataLoader(threadsPerBlock);
    mSolverParams.pCollisionDetection = new CollisionDetection{ this, _threadsPerBlockBVH, 1 << 16 };
    if (json.contains("dt")) {
        mSolverParams.dt = json["dt"].get<float>();
    }
    if (json.contains("gravity")) {
        mSolverParams.gravity = json["gravity"].get<float>();
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
            std::string nodeFile;
            std::string mshFile;
            std::string eleFile;
            if (sbDefJson.contains("nodeFile")) {
                nodeFile = sbDefJson["nodeFile"];
                eleFile = sbDefJson["eleFile"];
            }
            if (sbDefJson.contains("mshFile")) {
                mshFile = sbDefJson["mshFile"];
            }
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
                if (sbDefJson.contains("pos")) {
                    pos = glm::vec3(sbDefJson["pos"][0].get<float>(), sbDefJson["pos"][1].get<float>(), sbDefJson["pos"][2].get<float>());
                }
                else {
                    pos = glm::vec3(0.f);
                }
            }
            else {
                pos = glm::vec3(sbJson["pos"][0].get<float>(), sbJson["pos"][1].get<float>(), sbJson["pos"][2].get<float>());
            }
            if (!sbJson.contains("scale")) {
                if (sbDefJson.contains("scale")) {
                    scale = glm::vec3(sbDefJson["scale"][0].get<float>(), sbDefJson["scale"][1].get<float>(), sbDefJson["scale"][2].get<float>());
                }
                else {
                    scale = glm::vec3(1.f);
                }
            }
            else {
                scale = glm::vec3(sbJson["scale"][0].get<float>(), sbJson["scale"][1].get<float>(), sbJson["scale"][2].get<float>());
            }
            if (!sbJson.contains("rot")) {
                if (sbDefJson.contains("rot")) {
                    rot = glm::vec3(sbDefJson["rot"][0].get<float>(), sbDefJson["rot"][1].get<float>(), sbDefJson["rot"][2].get<float>());
                }
                else {
                    rot = glm::vec3(0.f);
                }
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
            if (!mshFile.empty()) {
                std::string baseName = mshFile.substr(nodeFile.find_last_of('/') + 1);
                char* name = new char[baseName.size() + 1];
                strcpy(name, baseName.c_str());
                namesSoftBodies.push_back(name);
                dataLoader.CollectData(mshFile.c_str(), pos, scale, rot, centralize, startIndex,
                    SolverAttribute{ mass, stiffness_0, stiffness_1, constraints });
            }
            else if (!nodeFile.empty()) {
                std::string baseName = nodeFile.substr(nodeFile.find_last_of('/') + 1);
                char* name = new char[baseName.size() + 1];
                strcpy(name, baseName.c_str());
                namesSoftBodies.push_back(name);
                dataLoader.CollectData(nodeFile.c_str(), eleFile.c_str(), faceFile.c_str(), pos, scale, rot, centralize, startIndex,
                    SolverAttribute{ mass, stiffness_0, stiffness_1, constraints });
            }
            else {
                throw std::runtime_error("Msh or node file must be provided!!!");
            }

        }
        dataLoader.AllocData(startIndices, mSolverData.X, mSolverData.X0, mSolverData.XTilde, mSolverData.V, mSolverData.Force, dev_Edges, mSolverData.Tet, dev_TetFathers, mSolverData.numVerts, mSolverData.numTets);
        for (auto softBodyData : dataLoader.m_softBodyData) {
            softBodies.push_back(new SoftBody(this, std::get<2>(softBodyData), &std::get<1>(softBodyData)));
        }
        mSolverParams.pCollisionDetection->Init(mSolverData.numTets, mSolverData.numVerts, maxThreads);
        cudaMalloc((void**)&mSolverData.dev_Normals, mSolverData.numVerts * sizeof(glm::vec3));
        cudaMalloc((void**)&mSolverData.dev_tIs, mSolverData.numVerts * sizeof(dataType));
    }
    mSolverData.pFixedBodies = new FixedBodyData{ _threadsPerBlock, _fixedBodies };
    mSolver = new PdSolver{ threadsPerBlock, mSolverData };
}

void SimulationCUDAContext::Update()
{
    mSolverParams.handleCollision = (context->guiData->handleCollision && context->guiData->BVHEnabled);
    mSolver->Update(mSolverData, mSolverParams);
    if (context->guiData->handleCollision || context->guiData->BVHEnabled) {
        mSolverParams.pCollisionDetection->PrepareRenderData();
    }
    if (context->guiData->ObjectVis) {
        PrepareRenderData();
    }
}


void SimulationCUDAContext::UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr) {
    softBodies[index]->SetAttributes(softBodyAttr);
}

void SimulationCUDAContext::SetBVHBuildType(BVH::BuildType buildType)
{
    mSolverParams.pCollisionDetection->SetBuildType(buildType);
}

void SimulationCUDAContext::SetGlobalSolver(bool useEigen)
{
    PdSolver* pdsolver = nullptr;
    if (pdsolver = dynamic_cast<PdSolver*>(mSolver)) {
        pdsolver->SetGlobalSolver(useEigen);
    }
}

void SimulationCUDAContext::Reset()
{
    cudaMemcpy(mSolverData.X, mSolverData.X0, sizeof(glm::vec3) * mSolverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(mSolverData.XTilde, mSolverData.X0, sizeof(glm::vec3) * mSolverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemset(mSolverData.V, 0, sizeof(glm::vec3) * mSolverData.numVerts);
    cudaMemset(mSolverData.Force, 0, sizeof(glm::vec3) * mSolverData.numVerts);
}

void SimulationCUDAContext::Draw(SurfaceShader* shaderProgram, SurfaceShader* flatShaderProgram)
{
    glLineWidth(2);
    if (context->guiData->ObjectVis) {
        shaderProgram->setModelMatrix(glm::mat4(1.f));
        for (auto softBody : softBodies)
            shaderProgram->draw(*softBody, 0);
        for (auto fixedBody : fixedBodies) {
            shaderProgram->setModelMatrix(fixedBody->m_model);
            shaderProgram->draw(*fixedBody, 0);
        }
    }
    if (context->guiData->handleCollision && context->guiData->BVHEnabled) 
        mSolverParams.pCollisionDetection->Draw(flatShaderProgram);
}

const SolverParams& SimulationCUDAContext::GetSolverParams() const
{
    return mSolverParams;
}
