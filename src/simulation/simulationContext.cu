#include <utilities.cuh>
#include <collision/bvh.h>
#include <simulation/solver/IPC/ipc.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/softBody.h>
#include <simulation/dataLoader.h>
#include <simulation/simulationContext.h>
#include <context.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>& _fixedBodies, int _threadsPerBlock, int _threadsPerBlockBVH, int maxThreads, int _numIterations)
    :context(ctx), threadsPerBlock(_threadsPerBlock), fixedBodies(_fixedBodies), name(_name)
{
    DataLoader<solverPrecision> dataLoader(threadsPerBlock);
    mSolverParams.pCollisionDetection = new CollisionDetection<solverPrecision>{ &mSolverData, ctx, _threadsPerBlockBVH, 1 << 16 };
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
            float mu;
            float lambda;
            std::vector<indexType> DBC;
            indexType* host_DBC;
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
            if (!sbJson.contains("mu")) {
                mu = sbDefJson["mu"].get<float>();
            }
            else {
                mu = sbJson["mu"].get<float>();
            }
            if (!sbJson.contains("lambda")) {
                lambda = sbDefJson["lambda"].get<float>();
            }
            else {
                lambda = sbJson["lambda"].get<float>();
            }
            if (!sbJson.contains("DBC")) {
                for (auto dbc : sbDefJson["DBC"]) {
                    DBC.push_back(dbc.get<int>());
                }
            }
            else {
                for (auto dbc : sbJson["DBC"]) {
                    DBC.push_back(dbc.get<int>());
                }
            }
            if (!DBC.empty()) {
                host_DBC = new indexType[DBC.size()];
                std::copy(DBC.begin(), DBC.end(), host_DBC);
            }
            else {
                host_DBC = nullptr;
            }
            bool centralize = sbDefJson["centralize"].get<bool>();
            int startIndex = sbDefJson["start index"].get<int>();
            SoftBodyAttribute attr{ mass, mu, lambda, host_DBC, DBC.size() };
            if (!mshFile.empty()) {
                std::string baseName = mshFile.substr(nodeFile.find_last_of('/') + 1);
                char* name = new char[baseName.size() + 1];
                strcpy(name, baseName.c_str());
                namesSoftBodies.push_back(name);
                dataLoader.CollectData(mshFile.c_str(), pos, scale, rot, centralize, startIndex, &attr);
            }
            else if (!nodeFile.empty()) {
                std::string baseName = nodeFile.substr(nodeFile.find_last_of('/') + 1);
                char* name = new char[baseName.size() + 1];
                strcpy(name, baseName.c_str());
                namesSoftBodies.push_back(name);
                dataLoader.CollectData(nodeFile.c_str(), eleFile.c_str(), faceFile.c_str(), pos, scale, rot, centralize, startIndex, &attr);
            }
            else {
                throw std::runtime_error("Msh or node file must be provided!!!");
            }

        }
        dataLoader.AllocData(startIndices, mSolverData, softBodies);
        mSolverParams.pCollisionDetection->Init(mSolverData.numTris, mSolverData.numVerts, maxThreads);
        cudaMalloc((void**)&mSolverData.dev_Normals, mSolverData.numVerts * sizeof(glm::vec3));
        cudaMalloc((void**)&mSolverData.dev_tIs, mSolverData.numVerts * sizeof(solverPrecision));
    }
    mSolverData.pFixedBodies = new FixedBodyData{ _threadsPerBlock, _fixedBodies };
    mSolver = new PdSolver{ threadsPerBlock, mSolverData };
}

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto name : namesSoftBodies) {
        delete[]name;
    }
    cudaFree(mSolverData.X);
    cudaFree(mSolverData.Tet);
    cudaFree(mSolverData.V);
    cudaFree(mSolverData.Force);
    cudaFree(mSolverData.X0);
    cudaFree(mSolverData.XTilde);
    cudaFree(mSolverData.ExtForce);
    cudaFree(mSolverData.DBC);
    cudaFree(mSolverData.mass);
    cudaFree(mSolverData.mu);
    cudaFree(mSolverData.lambda);
    cudaFree(mSolverData.dev_Edges);
    cudaFree(mSolverData.dev_TriFathers);

    for (auto softbody : softBodies) {
        delete softbody;
    }
    cudaFree(mSolverData.dev_Normals);
    delete mSolver;
}

int SimulationCUDAContext::GetVertCnt() const {
    return mSolverData.numVerts;
}

int SimulationCUDAContext::GetNumQueries() const {
    return mSolverParams.pCollisionDetection->GetNumQueries();
}

int SimulationCUDAContext::GetTetCnt() const {
    return mSolverData.numTets;
}

void SimulationCUDAContext::PrepareRenderData() {
    for (auto softbody : softBodies) {
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->Mesh::MapDevicePtr(&pos, &nor);
        dim3 numThreadsPerBlock(softbody->GetNumTris() / threadsPerBlock + 1);
        PopulateTriPos << <numThreadsPerBlock, threadsPerBlock >> > (pos, mSolverData.X, softbody->GetSoftBodyData().Tri, softbody->GetNumTris());
        RecalculateNormals << <softbody->GetNumTris() / threadsPerBlock + 1, threadsPerBlock >> > (nor, pos, softbody->GetNumTris());
        softbody->Mesh::UnMapDevicePtr();
    }
}