
#include <softBody.h>
#include <simulation/dataLoader.h>
#include <utilities.cuh>
#include <glm/gtc/matrix_transform.hpp>
#include <simulation/MshLoader.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;
template<typename HighP>
struct DataLoader<HighP>::Impl {
    std::vector<std::tuple<SolverData<HighP>, SoftBodyData, SoftBodyAttribute>> m_softBodyData;
    std::vector<std::vector<indexType>> m_edges;
};

template<typename HighP>
DataLoader<HighP>::DataLoader(int _threadsPerBlock) :threadsPerBlock(_threadsPerBlock), m_impl(new Impl)
{
}


template<typename HighP>
DataLoader<HighP>::~DataLoader() = default;

template<typename HighP>
std::vector<indexType> DataLoader<HighP>::loadEleFile(const std::string& EleFilename, int startIndex, int& numTets)
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
    return Tet;
}

template<typename HighP>
std::vector<indexType> DataLoader<HighP>::loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris)
{
    std::string line;
    std::ifstream file(faceFilename);

    if (!file.is_open()) {
        // std::cerr << "Unable to open face file" << std::endl;
        return std::vector<indexType>();
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numTris;

    std::vector<indexType> Triangle(numTris * 3);

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


template<typename HighP>
std::vector<glm::tvec3<HighP>> DataLoader<HighP>::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
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
    std::vector<glm::tvec3<HighP>> X(numVerts);
    glm::tvec3<HighP> center(0.0f);

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

template<typename HighP>
void DataLoader<HighP>::CollectEdges(const std::vector<indexType>& triIdx) {
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

template<typename HighP>
void DataLoader<HighP>::CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SoftBodyAttribute* attrib)
{
    totalNumDBC += attrib->numDBC;
    SolverData<HighP> solverData;
    SoftBodyData softBodyData;
    auto vertices = loadNodeFile(nodeFileName, centralize, solverData.numVerts);
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<HighP>) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::tvec3<HighP>) * solverData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (solverData.X, model, solverData.numVerts);

    auto tetIdx = loadEleFile(eleFileName, startIndex, solverData.numTets);
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(solverData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
    auto triIdx = loadFaceFile(faceFileName, startIndex, softBodyData.numTris);
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

    m_impl->m_softBodyData.push_back({ solverData,softBodyData, *attrib });
}

template<typename HighP>
void DataLoader<HighP>::CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SoftBodyAttribute* attrib)
{
    totalNumDBC += attrib->numDBC;
    SolverData<HighP> solverData;
    SoftBodyData softBodyData;
    igl::MshLoader _loader(mshFileName);
    auto nodes = _loader.get_nodes();
    std::vector<float> vertices(nodes.size());
    solverData.numVerts = nodes.size() / 3;
    std::transform(nodes.begin(), nodes.end(), vertices.begin(), [](igl::MshLoader::Float f) {
        return static_cast<float>(f);
        });
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<HighP>) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::tvec3<HighP>) * solverData.numVerts, cudaMemcpyHostToDevice);

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

    m_impl->m_softBodyData.push_back({ solverData,softBodyData, *attrib });
}

template<typename HighP>
void DataLoader<HighP>::AllocData(std::vector<int>& startIndices, SolverData<HighP>& solverData, indexType*& gEdges, indexType*& gTetFather, std::vector<SoftBody*>& softbodies)
{
    solverData.numVerts = totalNumVerts;
    solverData.numTets = totalNumTets;
    solverData.numDBC = totalNumDBC;
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.X0, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.XTilde, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.V, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.ExtForce, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMemset(solverData.V, 0, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMemset(solverData.ExtForce, 0, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * totalNumTets * 4);
    if (totalNumDBC > 0) {
        cudaMalloc((void**)&solverData.DBC, sizeof(indexType) * totalNumDBC);
    }
    cudaMalloc((void**)&solverData.contact_area, sizeof(HighP) * totalNumVerts);
    cudaMalloc((void**)&solverData.mass, sizeof(HighP) * totalNumVerts);
    cudaMalloc((void**)&solverData.mu, sizeof(HighP) * totalNumTets);
    cudaMalloc((void**)&solverData.lambda, sizeof(HighP) * totalNumTets);
    cudaMalloc((void**)&gEdges, sizeof(indexType) * totalNumEdges * 2);
    cudaMalloc((void**)&gTetFather, sizeof(indexType) * totalNumTets);
    int vertOffset = 0, tetOffset = 0, edgeOffset = 0, dbcOffset = 0;
    for (int i = 0; i < m_impl->m_softBodyData.size(); i++)
    {
        auto& softBody = m_impl->m_softBodyData[i];
        startIndices.push_back(vertOffset);
        SolverData<HighP>& softBodySolverData = std::get<0>(softBody);
        SoftBodyData& softBodyData = std::get<1>(softBody);
        const SoftBodyAttribute& softBodyAttr = std::get<2>(softBody);
        cudaMemcpy(solverData.X + vertOffset, softBodySolverData.X, sizeof(glm::tvec3<HighP>) * softBodySolverData.numVerts, cudaMemcpyDeviceToDevice);
        if (totalNumDBC > 0 && softBodyAttr.numDBC > 0) {
            thrust::host_vector<indexType> hDBC(softBodyAttr.DBC, softBodyAttr.DBC + softBodyAttr.numDBC);
            thrust::device_vector<indexType> dDBC(softBodyAttr.numDBC);
            thrust::copy(hDBC.begin(), hDBC.end(), dDBC.begin());
            thrust::device_ptr<indexType> dDBCPtr(solverData.DBC + dbcOffset);
            thrust::transform(dDBC.begin(), dDBC.end(), dDBCPtr, [vertOffset] __device__(indexType x) {
                return x + vertOffset;
            });
        }
        thrust::transform(softBodySolverData.Tet, softBodySolverData.Tet + softBodySolverData.numTets * 4, thrust::device_pointer_cast(solverData.Tet) + tetOffset, [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        if (softBodyData.Tri) {
            thrust::for_each(thrust::device_pointer_cast(softBodyData.Tri), thrust::device_pointer_cast(softBodyData.Tri) + softBodyData.numTris * 3, [vertOffset] __device__(indexType & x) {
                x += vertOffset;
            });
        }
        thrust::fill(thrust::device_pointer_cast(gTetFather) + tetOffset / 4, thrust::device_pointer_cast(gTetFather) + tetOffset / 4 + softBodySolverData.numTets, i);
        thrust::fill(thrust::device_pointer_cast(solverData.mass) + vertOffset, thrust::device_pointer_cast(solverData.mass) + vertOffset + softBodySolverData.numVerts, softBodyAttr.mass);
        thrust::fill(thrust::device_pointer_cast(solverData.mu) + tetOffset / 4, thrust::device_pointer_cast(solverData.mu) + tetOffset / 4 + softBodySolverData.numTets, softBodyAttr.mu);
        thrust::fill(thrust::device_pointer_cast(solverData.lambda) + tetOffset / 4, thrust::device_pointer_cast(solverData.lambda) + tetOffset / 4 + softBodySolverData.numTets, softBodyAttr.lambda);
        cudaMemcpy(gEdges + edgeOffset, m_impl->m_edges[i].data(), sizeof(indexType) * m_impl->m_edges[i].size(), cudaMemcpyHostToDevice);
        thrust::transform(thrust::device_pointer_cast(gEdges) + edgeOffset, thrust::device_pointer_cast(gEdges) + edgeOffset + m_impl->m_edges[i].size(), thrust::device_pointer_cast(gEdges) + edgeOffset,
            [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        cudaFree(softBodySolverData.X);
        cudaFree(softBodySolverData.Tet);
        std::get<1>(softBody).numTets = softBodySolverData.numTets;
        std::get<1>(softBody).Tet = solverData.Tet + tetOffset;
        vertOffset += softBodySolverData.numVerts;
        tetOffset += softBodySolverData.numTets * 4;
        edgeOffset += m_impl->m_edges[i].size();
        dbcOffset += softBodyAttr.numDBC;
        delete[] softBodyAttr.DBC;
        softbodies.push_back(new SoftBody(&softBodyData, softBodyAttr));
    }
    cudaMemcpy(solverData.X0, solverData.X, sizeof(glm::tvec3<HighP>) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.XTilde, solverData.X, sizeof(glm::tvec3<HighP>) * totalNumVerts, cudaMemcpyDeviceToDevice);
}

template class DataLoader<double>;
template class DataLoader<float>;