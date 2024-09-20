#pragma once

#include <mesh.h>
#include <simulation/solver/solver.h>
#include <vector>

class SoftBodyAttr;

class SoftBody : public Mesh {
public:
    SoftBody(const SoftBodyData* dataPtr, const SoftBodyAttribute _attrib, int threadsPerBlock = 256);
    ~SoftBody();
    SoftBody(const SoftBody&) = delete;
    SoftBody& operator=(const SoftBody&) = delete;
    int GetNumTris()const;
    const SoftBodyData& GetSoftBodyData() const;
    void SetAttributes(SoftBodyAttr* pSoftBodyAttr);
private:
    SoftBodyData softBodyData;
    SoftBodyAttribute attrib;
    bool jump = false;
    const int threadsPerBlock;
};
