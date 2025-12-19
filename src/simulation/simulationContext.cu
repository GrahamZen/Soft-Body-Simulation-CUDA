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

template <typename T>
void getJsonVal(const nlohmann::json& j, const char* key, T& target) {
    if (j.contains(key)) target = j[key].get<T>();
}

void getJsonVec3(const nlohmann::json& j, const char* key, const nlohmann::json& defJ, const char* defKey, glm::vec3& target) {
    if (j.contains(key))
        target = glm::vec3(j[key][0], j[key][1], j[key][2]);
    else if (defJ.contains(defKey))
        target = glm::vec3(defJ[defKey][0], defJ[defKey][1], defJ[defKey][2]);
    else
        target = glm::vec3(0.f);
}

template<class Scalar>
void SimulationCUDAContext::Impl<Scalar>::Init(Context* ctx, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>& _fixedBodies, int _threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations)
{
    threadsPerBlock = _threadsPerBlock;
    threadsPerBlockBVH = _threadsPerBlockBVH;
    maxThreads = _maxThreads;
    numIterations = _numIterations;

    auto guiData = ctx->guiData.get();
    DataLoader<Scalar> dataLoader(threadsPerBlock);
    std::vector<const char*> namesSoftBodies;
    data.pCollisionDetection = new CollisionDetection<Scalar>{ ctx, _threadsPerBlockBVH, 1 << 16 };
    params.numIterations = _numIterations;
    getJsonVal(json, "dt", params.dt);
    getJsonVal(json, "kappa", data.kappa);
    getJsonVal(json, "tolerance", params.tol);
    getJsonVal(json, "maxIterations", params.maxIterations);
    getJsonVal(json, "pauseIter", guiData->PauseIter);
    getJsonVal(json, "dhat", params.dhat);
    getJsonVal(json, "gravity", params.gravity);
    getJsonVal(json, "pause", guiData->Pause);
    getJsonVal(json, "damp", params.damp);
    getJsonVal(json, "muN", params.muN);
    getJsonVal(json, "muT", params.muT);
    if (json.contains("softBodies")) {
        for (const auto& sbJson : json["softBodies"]) {
            auto& sbDefJson = softBodyDefs.at(std::string(sbJson["name"]));
            std::string nodeFile = sbDefJson.value("nodeFile", "");
            std::string eleFile = sbDefJson.value("eleFile", "");
            std::string mshFile = sbDefJson.value("mshFile", "");
            std::string faceFile = sbDefJson.value("faceFile", "");
            glm::vec3 pos;
            glm::vec3 scale;
            glm::vec3 rot;
            float mass;
            float mu;
            float lambda;
            getJsonVec3(sbJson, "pos", sbDefJson, "pos", pos);
            getJsonVec3(sbJson, "scale", sbDefJson, "scale", scale);
            getJsonVec3(sbJson, "rot", sbDefJson, "rot", rot);
            mass = sbJson.value("mass", sbDefJson.value("mass", 1.0f));
            mu = sbJson.value("mu", sbDefJson.value("mu", 1000.0f));
            lambda = sbJson.value("lambda", sbDefJson.value("lambda", 1000.0f));
            std::vector<indexType> DBC;
            const auto& dbcSource = sbJson.contains("DBC") ? sbJson["DBC"] : sbDefJson["DBC"];
            if (!dbcSource.is_null()) {
                for (auto& dbc : dbcSource) DBC.push_back(dbc.get<int>());
            }

            indexType* host_DBC = DBC.empty() ? nullptr : new indexType[DBC.size()];
            if (host_DBC) std::copy(DBC.begin(), DBC.end(), host_DBC);
            bool centralize = sbDefJson.value("centralize", false);
            int startIndex = sbDefJson.value("start index", 0);
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
            if (impl.data.numTris <= 0 || impl.data.Tri == nullptr || impl.data.X == nullptr) 
                return false;
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
    if (i >= number) return;
    Scalar stiffness = 0;

    if (fixed[i] == 0 && select_v != -1)
    {
        glm::tvec3<Scalar> diff = X[i] - X[select_v];
        offset_X[i] = diff;
        Scalar dist2 = glm::dot(diff, diff);
        if (dist2 < RADIUS_SQUARED) {
            stiffness = control_mag;
        }
    }
    more_fixed[i] = stiffness;
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
