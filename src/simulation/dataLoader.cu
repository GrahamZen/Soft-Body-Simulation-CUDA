
#include <softBody.h>
#include <simulation/dataLoader.h>
#include <glm/gtc/matrix_transform.hpp>
#include <simulation/MshLoader.h>
#include <utilities.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <set>

using FaceVertIdx = std::tuple<indexType, indexType, indexType>;

template<typename T>
__host__ __device__ void sortThree(T& a, T& b, T& c);

namespace fs = std::filesystem;
template<typename Scalar>
struct DataLoader<Scalar>::Impl {
    std::vector<std::tuple<SolverData<Scalar>, SoftBodyData, SoftBodyAttribute>> m_softBodyData;
    std::vector<std::vector<indexType>> m_edges;
};

template<typename Scalar>
DataLoader<Scalar>::DataLoader(int _threadsPerBlock) :threadsPerBlock(_threadsPerBlock), m_impl(new Impl)
{
}


template<typename Scalar>
DataLoader<Scalar>::~DataLoader() = default;

template<typename Scalar>
std::pair<std::vector<indexType>, std::vector<indexType>> DataLoader<Scalar>::loadEleFaceFile(const std::string& EleFilename, int startIndex, int& numTets, int& numTris, std::string faceFilename)
{
    numTets = 0;
    numTris = 0;
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        fs::path absolutePath = fs::absolute(EleFilename);
        std::cerr << "Unable to open file: " << absolutePath << std::endl;
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numTets;
    std::vector<indexType> Tet(numTets * 4);

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

    std::vector<indexType> Triangle;
    if (!faceFilename.empty()) {
        std::string line;
        std::ifstream file(faceFilename);

        if (file.is_open()) {
            std::getline(file, line);
            std::istringstream iss(line);
            iss >> numTris;
            Triangle.resize(numTris * 3);

            int a, b, c, d, e;
            for (int tet = 0; tet < numTris && std::getline(file, line); ++tet) {
                std::istringstream iss(line);
                iss >> a >> b >> c >> d >> e;

                Triangle[tet * 3 + 0] = b - startIndex;
                Triangle[tet * 3 + 1] = c - startIndex;
                Triangle[tet * 3 + 2] = d - startIndex;
            }

            file.close();
        }
    }
    if (numTris == 0) {
        std::map<FaceVertIdx, FaceVertIdx> faceMap;
        std::set<FaceVertIdx> uniqueFaces;
        std::vector<FaceVertIdx> faces(4);
        for (size_t i = 0; i < Tet.size(); i += 4) {
            indexType v0 = Tet[i];
            indexType v1 = Tet[i + 1];
            indexType v2 = Tet[i + 2];
            indexType v3 = Tet[i + 3];
            faces[0] = std::make_tuple(v0, v1, v2);
            faces[1] = std::make_tuple(v0, v2, v3);
            faces[2] = std::make_tuple(v0, v3, v1);
            faces[3] = std::make_tuple(v1, v3, v2);
            auto tmpFaces = faces;
            // store the correct order of vertices for each face
            for (int j = 0; j < 4; ++j) {
                sortThree(std::get<0>(tmpFaces[j]), std::get<1>(tmpFaces[j]), std::get<2>(tmpFaces[j]));
                faceMap[tmpFaces[j]] = faces[j];
            }
            for (const auto& face : tmpFaces) {
                if (uniqueFaces.find(face) == uniqueFaces.end())
                    uniqueFaces.insert(face);
                else
                    uniqueFaces.erase(face);
            }
        }
        numTris = uniqueFaces.size();
        Triangle.resize(numTris * 3);
        int i = 0;
        for (const auto& face : uniqueFaces) {
            FaceVertIdx orderedFace = faceMap[face];
            Triangle[i++] = std::get<0>(orderedFace);
            Triangle[i++] = std::get<1>(orderedFace);
            Triangle[i++] = std::get<2>(orderedFace);
        }
    }
    return { Tet, Triangle };
}

template<typename Scalar>
std::vector<glm::tvec3<Scalar>> DataLoader<Scalar>::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
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
    std::vector<glm::tvec3<Scalar>> X(numVerts);
    glm::tvec3<Scalar> center(0.0f);

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

template<typename Scalar>
void DataLoader<Scalar>::CollectEdges(const std::vector<indexType>& triIdx) {
    std::set<std::pair<indexType, indexType>> uniqueEdges;
    std::vector<indexType> edges;

    for (size_t i = 0; i < triIdx.size(); i += 3) {
        indexType v0 = triIdx[i];
        indexType v1 = triIdx[i + 1];
        indexType v2 = triIdx[i + 2];

        std::pair<indexType, indexType> edge1 = std::minmax(v0, v1);
        std::pair<indexType, indexType> edge2 = std::minmax(v1, v2);
        std::pair<indexType, indexType> edge3 = std::minmax(v2, v0);

        uniqueEdges.insert(edge1);
        uniqueEdges.insert(edge2);
        uniqueEdges.insert(edge3);
    }

    for (const auto& edge : uniqueEdges) {
        edges.push_back(edge.first);
        edges.push_back(edge.second);
    }

    m_impl->m_edges.push_back(edges);
    totalNumEdges += edges.size() / 2;
}

template<typename Scalar>
void DataLoader<Scalar>::CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SoftBodyAttribute* attrib)
{
    totalNumDBC += attrib->numDBC;
    SolverData<Scalar> solverData;
    SoftBodyData softBodyData;
    auto vertices = loadNodeFile(nodeFileName, centralize, solverData.numVerts);
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<Scalar>) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::tvec3<Scalar>) * solverData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (solverData.X, model, solverData.numVerts);

    auto [tetIdx, triIdx] = loadEleFaceFile(eleFileName, startIndex, solverData.numTets, softBodyData.numTris, faceFileName);
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(solverData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
    if (softBodyData.numTris != 0) {
        cudaMalloc((void**)&softBodyData.Tri, sizeof(indexType) * triIdx.size());
        cudaMemcpy(softBodyData.Tri, triIdx.data(), sizeof(indexType) * triIdx.size(), cudaMemcpyHostToDevice);
    }
    CollectEdges(triIdx);
    totalNumVerts += solverData.numVerts;
    totalNumTets += solverData.numTets;
    totalNumTris += softBodyData.numTris;

    m_impl->m_softBodyData.push_back({ solverData,softBodyData, *attrib });
}

template<typename Scalar>
void DataLoader<Scalar>::CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SoftBodyAttribute* attrib)
{
    totalNumDBC += attrib->numDBC;
    SolverData<Scalar> solverData;
    SoftBodyData softBodyData;
    igl::MshLoader _loader(mshFileName);
    auto nodes = _loader.get_nodes();
    std::vector<float> vertices(nodes.size());
    solverData.numVerts = nodes.size() / 3;
    std::transform(nodes.begin(), nodes.end(), vertices.begin(), [](igl::MshLoader::Float f) {
        return static_cast<float>(f);
        });
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<Scalar>) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::tvec3<Scalar>) * solverData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (solverData.X, model, solverData.numVerts);

    auto elements = _loader.get_elements();
    std::vector<indexType> tetIdx(elements.size());
    std::transform(elements.begin(), elements.end(), tetIdx.begin(), [](int i) {
        return static_cast<indexType>(i);
        });
    solverData.numTets = tetIdx.size() / 4;
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(solverData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
    std::vector<indexType> triIdx;
    if (!triIdx.empty()) {
        cudaMalloc((void**)&softBodyData.Tri, sizeof(indexType) * triIdx.size());
        cudaMemcpy(softBodyData.Tri, triIdx.data(), sizeof(indexType) * triIdx.size(), cudaMemcpyHostToDevice);
    }
    else {
        softBodyData.Tri = nullptr;
        softBodyData.numTris = 0;
    }
    CollectEdges(triIdx);
    totalNumVerts += solverData.numVerts;
    totalNumTets += solverData.numTets;
    totalNumTris += softBodyData.numTris;

    m_impl->m_softBodyData.push_back({ solverData,softBodyData, *attrib });
}

template<typename Scalar>
void DataLoader<Scalar>::AllocData(std::vector<int>& startIndices, SolverData<Scalar>& solverData, std::vector<SoftBody*>& softbodies, const std::vector<const char*>& namesSoftBodies)
{
    solverData.numVerts = totalNumVerts;
    solverData.numTets = totalNumTets;
    solverData.numTris = totalNumTris;
    solverData.numDBC = totalNumDBC;
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMalloc((void**)&solverData.X0, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMalloc((void**)&solverData.XTilde, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMalloc((void**)&solverData.V, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMalloc((void**)&solverData.ExtForce, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMalloc((void**)&solverData.OffsetX, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMalloc((void**)&solverData.DBC, sizeof(Scalar) * totalNumVerts);
    cudaMalloc((void**)&solverData.moreDBC, sizeof(Scalar) * totalNumVerts);
    cudaMalloc((void**)&solverData.DBCX, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMemset(solverData.V, 0, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMemset(solverData.ExtForce, 0, sizeof(glm::tvec3<Scalar>) * totalNumVerts);
    cudaMemset(solverData.DBC, 0, sizeof(Scalar) * totalNumVerts);
    cudaMemset(solverData.moreDBC, 0, sizeof(Scalar) * totalNumVerts);
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * totalNumTets * 4);
    cudaMalloc((void**)&solverData.Tri, sizeof(indexType) * totalNumTris * 3);
    if (totalNumDBC > 0) {
        cudaMalloc((void**)&solverData.DBCIdx, sizeof(indexType) * totalNumDBC);
    }
    cudaMalloc((void**)&solverData.contact_area, sizeof(Scalar) * totalNumVerts);
    cudaMalloc((void**)&solverData.mass, sizeof(Scalar) * totalNumVerts);
    cudaMalloc((void**)&solverData.mu, sizeof(Scalar) * totalNumTets);
    cudaMalloc((void**)&solverData.lambda, sizeof(Scalar) * totalNumTets);
    cudaMalloc((void**)&solverData.dev_Edges, sizeof(indexType) * totalNumEdges * 2);
    cudaMalloc((void**)&solverData.dev_TriFathers, sizeof(indexType) * totalNumTris);
    int vertOffset = 0, tetOffset = 0, triOffset = 0, edgeOffset = 0, dbcOffset = 0;
    for (int i = 0; i < m_impl->m_softBodyData.size(); i++)
    {
        auto& softBody = m_impl->m_softBodyData[i];
        startIndices.push_back(vertOffset);
        SolverData<Scalar>& softBodySolverData = std::get<0>(softBody);
        SoftBodyData& softBodyData = std::get<1>(softBody);
        const SoftBodyAttribute& softBodyAttr = std::get<2>(softBody);
        cudaMemcpy(solverData.X + vertOffset, softBodySolverData.X, sizeof(glm::tvec3<Scalar>) * softBodySolverData.numVerts, cudaMemcpyDeviceToDevice);
        if (totalNumDBC > 0 && softBodyAttr.numDBC > 0) {
            thrust::host_vector<indexType> hDBC(softBodyAttr.DBC, softBodyAttr.DBC + softBodyAttr.numDBC);
            thrust::device_vector<indexType> dDBC(softBodyAttr.numDBC);
            thrust::copy(hDBC.begin(), hDBC.end(), dDBC.begin());
            thrust::device_ptr<indexType> dDBCPtr(solverData.DBCIdx + dbcOffset);
            thrust::transform(dDBC.begin(), dDBC.end(), dDBCPtr, [vertOffset] __device__(indexType x) {
                return x + vertOffset;
            });
            thrust::for_each(dDBC.begin(), dDBC.end(), [vertOffset, solverData] __device__(indexType x) {
                solverData.DBC[x + vertOffset] = 1;
            });
        }
        thrust::transform(softBodySolverData.Tet, softBodySolverData.Tet + softBodySolverData.numTets * 4, thrust::device_pointer_cast(solverData.Tet) + tetOffset, [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        if (softBodyData.Tri) {
            auto first = thrust::device_pointer_cast(softBodyData.Tri);
            auto last = first + softBodyData.numTris * 3;
            thrust::transform(first, last, first,
                [vertOffset] __device__(indexType x) {
                return x + vertOffset;
            }
            );
            cudaMemcpy(solverData.Tri + triOffset, softBodyData.Tri, sizeof(indexType) * softBodyData.numTris * 3, cudaMemcpyDeviceToDevice);
        }
        thrust::fill(thrust::device_pointer_cast(solverData.dev_TriFathers) + triOffset / 3, thrust::device_pointer_cast(solverData.dev_TriFathers) + triOffset / 3 + softBodyData.numTris, i);
        thrust::fill(thrust::device_pointer_cast(solverData.mass) + vertOffset, thrust::device_pointer_cast(solverData.mass) + vertOffset + softBodySolverData.numVerts, softBodyAttr.mass);
        thrust::fill(thrust::device_pointer_cast(solverData.mu) + tetOffset / 4, thrust::device_pointer_cast(solverData.mu) + tetOffset / 4 + softBodySolverData.numTets, softBodyAttr.mu);
        thrust::fill(thrust::device_pointer_cast(solverData.lambda) + tetOffset / 4, thrust::device_pointer_cast(solverData.lambda) + tetOffset / 4 + softBodySolverData.numTets, softBodyAttr.lambda);
        cudaMemcpy(solverData.dev_Edges + edgeOffset, m_impl->m_edges[i].data(), sizeof(indexType) * m_impl->m_edges[i].size(), cudaMemcpyHostToDevice);
        thrust::transform(thrust::device_pointer_cast(solverData.dev_Edges) + edgeOffset, thrust::device_pointer_cast(solverData.dev_Edges) + edgeOffset + m_impl->m_edges[i].size(), thrust::device_pointer_cast(solverData.dev_Edges) + edgeOffset,
            [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        cudaFree(softBodySolverData.X);
        cudaFree(softBodySolverData.Tet);
        softbodies.push_back(new SoftBody(&softBodyData, softBodyAttr, { tetOffset / 4, tetOffset / 4 + softBodySolverData.numTets }, threadsPerBlock, namesSoftBodies[i]));
        vertOffset += softBodySolverData.numVerts;
        triOffset += softBodyData.numTris * 3;
        tetOffset += softBodySolverData.numTets * 4;
        edgeOffset += m_impl->m_edges[i].size();
        dbcOffset += softBodyAttr.numDBC;
        delete[] softBodyAttr.DBC;
    }
    cudaMemcpy(solverData.X0, solverData.X, sizeof(glm::tvec3<Scalar>) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.DBCX, solverData.X0, sizeof(glm::tvec3<Scalar>) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.XTilde, solverData.X, sizeof(glm::tvec3<Scalar>) * totalNumVerts, cudaMemcpyDeviceToDevice);
}

template<typename Scalar>
void DataLoader<Scalar>::FillData(Scalar* X, Scalar val, indexType* Tet, std::pair<size_t, size_t> tetIdxRange)
{
    thrust::fill(thrust::device_pointer_cast(X) + tetIdxRange.first, thrust::device_pointer_cast(X) + tetIdxRange.second, val);
}

template class DataLoader<double>;
template class DataLoader<float>;