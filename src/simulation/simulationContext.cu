#include <utilities.cuh>
#include <collision/bvh.h>
#include <collision/intersections.h>
#include <simulation/solver/IPC/ipc.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/softBody.h>
#include <simulation/dataLoader.h>
#include <simulation/simulationContext.h>
#include <context.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <type_traits>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define RADIUS_SQUARED		0.002

template<class Scalar>
void SimulationCUDAContext::Impl<Scalar>::Init(Context* ctx, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>& _fixedBodies, int _threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations)
{
    threadsPerBlock = _threadsPerBlock;
    threadsPerBlockBVH = _threadsPerBlockBVH;
    maxThreads = _maxThreads;
    numIterations = _numIterations;

    auto guiData = ctx->guiData;
    DataLoader<Scalar> dataLoader(threadsPerBlock);
    std::vector<const char*> namesSoftBodies;
    data.pCollisionDetection = new CollisionDetection<Scalar>{ ctx, _threadsPerBlockBVH, 1 << 16 };
    params.numIterations = _numIterations;
    if (json.contains("dt")) {
        params.dt = json["dt"].get<Scalar>();
    }
    if (json.contains("kappa")) {
        data.kappa = json["kappa"].get<Scalar>();
    }
    if (json.contains("tolerance")) {
        params.tol = json["tolerance"].get<Scalar>();
    }
    if (json.contains("maxIterations")) {
        params.maxIterations = json["maxIterations"].get<int>();
    }
    if (json.contains("pauseIter")) {
        guiData->PauseIter = json["pauseIter"].get<int>();
    }
    if (json.contains("dhat")) {
        params.dhat = json["dhat"].get<Scalar>();
    }
    if (json.contains("gravity")) {
        params.gravity = json["gravity"].get<Scalar>();
    }
    if (json.contains("pause")) {
        guiData->Pause = json["pause"].get<bool>();
    }
    if (json.contains("damp")) {
        params.damp = json["damp"].get<Scalar>();
    }
    if (json.contains("muN")) {
        params.muN = json["muN"].get<Scalar>();
    }
    if (json.contains("muT")) {
        params.muT = json["muT"].get<Scalar>();
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
        dataLoader.AllocData(startIndices, data, softBodies, namesSoftBodies);
        data.pCollisionDetection->Init(data.numTris, data.numVerts, maxThreads);
        cudaMalloc((void**)&data.dev_Normals, data.numVerts * sizeof(glm::vec3));
        cudaMalloc((void**)&data.dev_tIs, data.numVerts * sizeof(Scalar));
    }
    fixedBodies = _fixedBodies;
    data.pFixedBodies = new FixedBodyData{ _threadsPerBlock, _fixedBodies };

    if constexpr (std::is_same_v<Scalar, double>) {
        solver = std::make_unique<IPCSolver>(threadsPerBlock, data);
    }
    else {
        solver = std::make_unique<PdSolver>(threadsPerBlock, data);
    }
    solver->SetPerf(true);
}

template<class Scalar>
SimulationCUDAContext::Impl<Scalar>::~Impl()
{
    cudaFree(data.X);
    cudaFree(data.Tet);
    cudaFree(data.V);
    cudaFree(data.Force);
    cudaFree(data.X0);
    cudaFree(data.XTilde);
    cudaFree(data.ExtForce);
    cudaFree(data.DBC);
    cudaFree(data.mass);
    cudaFree(data.mu);
    cudaFree(data.lambda);
    cudaFree(data.dev_Edges);
    cudaFree(data.dev_TriFathers);
    cudaFree(data.dev_Normals);
    cudaFree(data.dev_tIs);

    for (auto softbody : softBodies) {
        delete softbody;
    }
    delete data.pCollisionDetection;
}

void SimulationCUDAContext::UpdateSoftBodyAttr(int index, SoftBodyAttr* pSoftBodyAttr)
{
    VisitImpl([&](auto& impl) {
        using Scalar = typename std::decay_t<decltype(impl)>::ScalarType;
        if (pSoftBodyAttr->mu) {
            DataLoader<Scalar>::FillData(impl.data.mu, impl.softBodies[index]->GetAttributes().mu, impl.data.Tet, impl.softBodies[index]->GetTetIdxRange());
        }
        if (pSoftBodyAttr->lambda) {
            DataLoader<Scalar>::FillData(impl.data.lambda, impl.softBodies[index]->GetAttributes().lambda, impl.data.Tet, impl.softBodies[index]->GetTetIdxRange());
        }
        });
}
bool SimulationCUDAContext::RayIntersect(const Ray& ray, glm::vec3* pos, bool updateV)
{
    return VisitImpl([&](auto& impl) {
        auto& ms = impl.data.mouseSelection;
        indexType hit_v = -1;
        if (ms.dragging && ms.select_v != -1) {
            hit_v = ms.select_v;
        }
        else {
            hit_v = raySimCtxIntersection(ray, impl.data.numTris, impl.data.Tri, impl.data.X);
            if (updateV) ms.select_v = hit_v;
        }
        if (hit_v == static_cast<indexType>(-1)) return false;
        glm::vec3 p;
        cudaMemcpy(&p, impl.data.X + hit_v, sizeof(p), cudaMemcpyDeviceToHost);
        if (pos) *pos = p;
        float dist = glm::dot(p - ray.origin, ray.direction);
        ms.target = ray.origin + dist * ray.direction;
        return true;
        });
}

template<class Scalar>
__global__ void Control_Kernel(glm::tvec3<Scalar>* X, Scalar* fixed, Scalar* more_fixed, glm::tvec3<Scalar>* offset_X, const Scalar control_mag, const int number, const int select_v)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= number)	return;

    more_fixed[i] = 0;
    if (fixed[i] == 0 && select_v != -1)
    {
        offset_X[i].x = X[i].x - X[select_v].x;
        offset_X[i].y = X[i].y - X[select_v].y;
        offset_X[i].z = X[i].z - X[select_v].z;

        Scalar dist2 = offset_X[i].x * offset_X[i].x + offset_X[i].y * offset_X[i].y + offset_X[i].z * offset_X[i].z;
        if (dist2 < RADIUS_SQUARED)	more_fixed[i] = control_mag;
    }
}

void SimulationCUDAContext::ResetMoreDBC(bool clear)
{
    VisitImpl([&](auto& impl) {
        using Scalar = typename std::decay_t<decltype(impl)>::ScalarType;
        if (clear) {
            cudaMemset(impl.data.moreDBC, 0, impl.data.numVerts * sizeof(Scalar));
            return;
        }
        if (impl.data.mouseSelection.dragging)
            Control_Kernel << <impl.data.numVerts / impl.threadsPerBlock + 1, impl.threadsPerBlock >> > (impl.data.X, impl.data.DBC, impl.data.moreDBC, impl.data.OffsetX, static_cast<Scalar>(10), impl.data.numVerts, impl.data.mouseSelection.select_v);
        });
}

void SimulationCUDAContext::Reset()
{
    VisitImpl([&](auto& impl) {
        using Scalar = typename std::decay_t<decltype(impl)>::ScalarType;
        cudaMemcpy(impl.data.X, impl.data.X0, sizeof(glm::tvec3<Scalar>) * impl.data.numVerts, cudaMemcpyDeviceToDevice);
        cudaMemcpy(impl.data.XTilde, impl.data.X0, sizeof(glm::tvec3<Scalar>) * impl.data.numVerts, cudaMemcpyDeviceToDevice);
        cudaMemset(impl.data.V, 0, sizeof(glm::tvec3<Scalar>) * impl.data.numVerts);
        cudaMemset(impl.data.moreDBC, 0, sizeof(Scalar) * impl.data.numVerts);
        impl.solver->Reset();
        });
}

void SimulationCUDAContext::PrepareRenderData() {
    VisitImpl([&](auto& impl) {
        for (auto softbody : impl.softBodies) {
            glm::vec3* pos;
            glm::vec4* nor;
            softbody->Mesh::MapDevicePtr(&pos, &nor);
            dim3 numThreadsPerBlock(softbody->GetNumTris() / impl.threadsPerBlock + 1);
            PopulateTriPos << <numThreadsPerBlock, impl.threadsPerBlock >> > (pos, impl.data.X, softbody->GetSoftBodyData().Tri, softbody->GetNumTris());
            RecalculateNormals << <softbody->GetNumTris() / impl.threadsPerBlock + 1, impl.threadsPerBlock >> > (nor, pos, softbody->GetNumTris());
            softbody->Mesh::UnMapDevicePtr();
        }
        });
}

template struct SimulationCUDAContext::Impl<float>;
template struct SimulationCUDAContext::Impl<double>;