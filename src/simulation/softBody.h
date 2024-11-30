#pragma once

#include <mesh.h>
#include <simulation/solver/solver.h>
#include <vector>
#include <string>

class SoftBodyAttr;

class SoftBody : public Mesh {
public:
    SoftBody(const SoftBodyData* dataPtr, const SoftBodyAttribute _attrib, std::pair<size_t, size_t> tetIdxRange, int threadsPerBlock = 256, const char* name = nullptr);
    ~SoftBody();
    SoftBody(const SoftBody&) = delete;
    SoftBody& operator=(const SoftBody&) = delete;
    int GetNumTris()const;
    std::string GetName()const { return name; }
    SoftBodyAttribute& GetAttributes() { return attrib; }
    std::pair<size_t, size_t> GetTetIdxRange() const { return tetIdxRange; }
    const SoftBodyData& GetSoftBodyData() const;
private:
    SoftBodyData softBodyData;
    SoftBodyAttribute attrib;
    bool jump = false;
    const int threadsPerBlock;
    const std::string name;
    const std::pair<size_t, size_t> tetIdxRange;
};
