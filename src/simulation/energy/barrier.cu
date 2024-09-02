#include <energy/barrier.h>
#include <solver/solverUtil.cuh>
#include <fixedBodyData.h>
#include <plane.h>
#include <cylinder.h>
#include <sphere.h>
#include <glm/glm.hpp>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
    template <typename HighP>
    __global__ void hessianKern(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx, glm::tvec3<HighP>* X, int numVerts,
        Plane* planes, int numPlanes, Cylinder* cylinders, int numCylinders, Sphere* spheres, int numSpheres,
        HighP dhat, HighP* contact_area, HighP coef) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts)
            return;
        glm::tvec3<HighP> x = X[idx];
        for (int j = 0; j < numPlanes; j++) {
            const Plane& plane = planes[j];
            glm::tvec3<HighP> floorPos = glm::tvec3<HighP>(plane.m_model[3]);
            glm::tvec3<HighP> floorUp = plane.m_floorUp;
            HighP d = glm::dot(x - floorPos, floorUp);
            if (d < dhat)
            {
                glm::tmat3x3<HighP> hess = coef * contact_area[idx] * dhat * plane.kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(floorUp, floorUp);
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        int rowIdx = idx * 3 + k;
                        int colIdx = idx * 3 + l;
                        int index = idx * 9 + k * 3 + l;
                        hessianVal[index] = hess[k][l];
                        hessianRowIdx[index] = rowIdx;
                        hessianColIdx[index] = colIdx;
                    }
                }
            }
        }
        for (int j = 0; j < numCylinders; j++) {
            const Cylinder cy = cylinders[j];
            glm::tvec3<HighP> axis = glm::tvec3<HighP>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<HighP> nnT = glm::tmat3x3<HighP>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<HighP> cylinderCenter = glm::tvec3<HighP>(cy.m_model[3]);
            HighP cylinderRadius = cy.m_radius;
            glm::tvec3<HighP> n = nnT * (x - cylinderCenter);
            HighP d = glm::length(n) - cylinderRadius;
            glm::tvec3<HighP> normal = glm::normalize(n);
            if (d < dhat)
            {
                glm::tmat3x3<HighP> hess = coef * contact_area[idx] * dhat * cy.kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(normal, normal);
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        int rowIdx = idx * 3 + k;
                        int colIdx = idx * 3 + l;
                        int index = idx * 9 + k * 3 + l;
                        hessianVal[index] += hess[k][l];
                        hessianRowIdx[index] = rowIdx;
                        hessianColIdx[index] = colIdx;
                    }
                }
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<HighP> sphereCenter = glm::tvec3<HighP>(sphere.m_model[3]);
            HighP sphereRadius = sphere.m_radius;
            glm::tvec3<HighP> n = x - sphereCenter;
            HighP d = glm::length(n) - sphereRadius;
            glm::tvec3<HighP> normal = glm::normalize(n);
            if (d < dhat)
            {
                glm::tmat3x3<HighP> hess = coef * contact_area[idx] * dhat * sphere.kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(normal, normal);
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        int rowIdx = idx * 3 + k;
                        int colIdx = idx * 3 + l;
                        int index = idx * 9 + k * 3 + l;
                        hessianVal[index] += hess[k][l];
                        hessianRowIdx[index] = rowIdx;
                        hessianColIdx[index] = colIdx;
                    }
                }
            }
        }
    }

    template <typename HighP>
    __global__ void gradientKern(HighP* grad, glm::tvec3<HighP>* X, int numVerts, Plane* planes, int numPlanes, Cylinder* cylinders, int numCylinders, Sphere* spheres, int numSpheres,
        HighP dhat, HighP* contact_area, HighP coef) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts)
            return;
        const glm::tvec3<HighP> x = X[idx];
        for (int j = 0; j < numPlanes; j++) {
            const Plane& plane = planes[j];
            glm::tvec3<HighP> floorPos = glm::tvec3<HighP>(plane.m_model[3]);
            glm::tvec3<HighP> floorUp = plane.m_floorUp;
            HighP d = glm::dot(x - floorPos, floorUp);
            if (d < dhat)
            {
                HighP s = d / dhat;
                glm::tvec3<HighP> gradient = coef * contact_area[idx] * dhat * (plane.kappa / 2 * (log(s) / dhat + (s - 1) / d)) * floorUp;
                grad[idx * 3] += gradient.x;
                grad[idx * 3 + 1] += gradient.y;
                grad[idx * 3 + 2] += gradient.z;
            }
        }
        for (int j = 0; j < numCylinders; j++) {
            const Cylinder cy = cylinders[j];
            glm::tvec3<HighP> axis = glm::tvec3<HighP>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<HighP> nnT = glm::tmat3x3<HighP>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<HighP> cylinderCenter = glm::tvec3<HighP>(cy.m_model[3]);
            HighP cylinderRadius = cy.m_radius;
            glm::tvec3<HighP> n = nnT * (x - cylinderCenter);
            HighP d = glm::length(n) - cylinderRadius;
            glm::tvec3<HighP> normal = glm::normalize(n);
            if (d < dhat)
            {
                HighP s = d / dhat;
                glm::tvec3<HighP> gradient = coef * contact_area[idx] * dhat * (cy.kappa / 2 * (log(s) / dhat + (s - 1) / d)) * normal;
                grad[idx * 3] += gradient.x;
                grad[idx * 3 + 1] += gradient.y;
                grad[idx * 3 + 2] += gradient.z;
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<HighP> sphereCenter = glm::tvec3<HighP>(sphere.m_model[3]);
            HighP sphereRadius = sphere.m_radius;
            glm::tvec3<HighP> n = x - sphereCenter;
            HighP d = glm::length(n) - sphereRadius;
            glm::tvec3<HighP> normal = glm::normalize(n);
            if (d < dhat)
            {
                HighP s = d / dhat;
                glm::tvec3<HighP> gradient = coef * contact_area[idx] * dhat * (sphere.kappa / 2 * (log(s) / dhat + (s - 1) / d)) * normal;
                grad[idx * 3] += gradient.x;
                grad[idx * 3 + 1] += gradient.y;
                grad[idx * 3 + 2] += gradient.z;
            }
        }
    }
}

template <typename HighP>
int BarrierEnergy<HighP>::NNZ(const SolverData<HighP>& solverData) const { return solverData.numVerts * 9; }

template <typename HighP>
BarrierEnergy<HighP>::BarrierEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset, HighP dhat) :dhat(dhat), Energy<HighP>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template <typename HighP>
HighP BarrierEnergy<HighP>::Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const {
    const Plane* planes = solverData.pFixedBodies->dev_planes;
    const Cylinder* cylinders = solverData.pFixedBodies->dev_cylinders;
    const Sphere* spheres = solverData.pFixedBodies->dev_spheres;
    int numSpheres = solverData.pFixedBodies->numSpheres;
    int numCylinders = solverData.pFixedBodies->numCylinders;
    int numPlanes = solverData.pFixedBodies->numPlanes;
    HighP dhat = this->dhat;
    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=]__host__ __device__(indexType vertIdx) {
        const glm::tvec3<HighP> x = Xs[vertIdx];
        HighP sum = 0.0;
        for (int j = 0; j < numPlanes; j++)
        {
            const Plane& plane = planes[j];
            glm::tvec3<HighP> floorPos = glm::tvec3<HighP>(plane.m_model[3]);
            glm::tvec3<HighP> floorUp = plane.m_floorUp;
            HighP d = glm::dot(x - floorPos, floorUp);
            if (d < dhat)
            {
                HighP s = d / dhat;
                sum += solverData.contact_area[vertIdx] * dhat * plane.kappa * 0.5 * (s - 1) * log(s);
            }
        }
        for (int j = 0; j < numCylinders; j++) {
            const Cylinder cy = cylinders[j];
            glm::tvec3<HighP> axis = glm::tvec3<HighP>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<HighP> nnT = glm::tmat3x3<HighP>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<HighP> cylinderCenter = glm::tvec3<HighP>(cy.m_model[3]);
            HighP cylinderRadius = cy.m_radius;
            glm::tvec3<HighP> n = nnT * (x - cylinderCenter);
            HighP d = glm::length(n) - cylinderRadius;
            if (d < dhat)
            {
                HighP s = d / dhat;
                sum += solverData.contact_area[vertIdx] * dhat * cy.kappa * 0.5 * (s - 1) * log(s);
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<HighP> sphereCenter = glm::tvec3<HighP>(sphere.m_model[3]);
            HighP sphereRadius = sphere.m_radius;
            glm::tvec3<HighP> n = x - sphereCenter;
            HighP d = glm::length(n) - sphereRadius;
            glm::tvec3<HighP> normal = glm::normalize(n);
            if (d < dhat)
            {
                HighP s = d / dhat;
                sum += solverData.contact_area[vertIdx] * dhat * sphere.kappa * 0.5 * (s - 1) * log(s);
            }
        }
        return sum;
    },
        0.0,
        thrust::plus<HighP>());
    return sum;
}

template<typename HighP>
void BarrierEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::gradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.numVerts, solverData.pFixedBodies->dev_planes, solverData.pFixedBodies->numPlanes,
        solverData.pFixedBodies->dev_cylinders, solverData.pFixedBodies->numCylinders, solverData.pFixedBodies->dev_spheres, solverData.pFixedBodies->numSpheres, dhat, solverData.contact_area, coef);
}

template <typename HighP>
void BarrierEnergy<HighP>::Hessian(const SolverData<HighP>& solverData, HighP coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::hessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx, solverData.X, solverData.numVerts,
        solverData.pFixedBodies->dev_planes, solverData.pFixedBodies->numPlanes, solverData.pFixedBodies->dev_cylinders, solverData.pFixedBodies->numCylinders, solverData.pFixedBodies->dev_spheres, solverData.pFixedBodies->numSpheres, dhat, solverData.contact_area, coef);
}

template<typename HighP>
HighP BarrierEnergy<HighP>::InitStepSize(const SolverData<HighP>& solverData, HighP* p) const
{
    const Plane* planes = solverData.pFixedBodies->dev_planes;
    const Cylinder* cylinders = solverData.pFixedBodies->dev_cylinders;
    const Sphere* spheres = solverData.pFixedBodies->dev_spheres;
    int numSpheres = solverData.pFixedBodies->numSpheres;
    int numCylinders = solverData.pFixedBodies->numCylinders;
    int numPlanes = solverData.pFixedBodies->numPlanes;
    return thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=]__host__ __device__(indexType vertIdx) {
        HighP alpha = 1.0;
        glm::tvec3<HighP> localP{ -p[vertIdx * 3], -p[vertIdx * 3 + 1], -p[vertIdx * 3 + 2] };
        const glm::tvec3<HighP> x = solverData.X[vertIdx];
        for (int j = 0; j < numPlanes; j++)
        {
            glm::tvec3<HighP> floorUp = planes[j].m_floorUp;
            glm::tvec3<HighP> floorPos = glm::tvec3<HighP>(planes[j].m_model[3]);
            HighP p_n = glm::dot(localP, floorUp);
            if (p_n < 0)
            {
                alpha = min(alpha, 0.9 * glm::dot(floorUp, x - floorPos) / -p_n);
            }
        }
        for (int j = 0; j < numCylinders; j++)
        {
            const Cylinder cy = cylinders[j];
            glm::tvec3<HighP> axis = glm::tvec3<HighP>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<HighP> nnT = glm::tmat3x3<HighP>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<HighP> cylinderCenter = glm::tvec3<HighP>(cy.m_model[3]);
            HighP cylinderRadius = cy.m_radius;
            glm::tvec3<HighP> n = nnT * (x - cylinderCenter);
            HighP p_n = glm::dot(localP, n);
            if (p_n < 0)
            {
                glm::tvec3<HighP> pp = nnT * localP;
                HighP pp2 = glm::dot(pp, pp);
                HighP n2 = glm::dot(n, n);
                HighP ndotpp = glm::dot(n, pp);
                alpha = min(alpha, 0.9 * (-ndotpp - sqrt(ndotpp * ndotpp - pp2 * (n2 - cylinderRadius * cylinderRadius))) / pp2);
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<HighP> sphereCenter = glm::tvec3<HighP>(sphere.m_model[3]);
            HighP sphereRadius = sphere.m_radius;
            glm::tvec3<HighP> n = x - sphereCenter;
            HighP p_n = glm::dot(localP, n);
            if (p_n < 0)
            {
                HighP ndotp = glm::dot(n, localP);
                HighP pp = glm::dot(localP, localP);
                HighP nn = glm::dot(n, n);
                alpha = min(alpha, 0.9 * (-ndotp - sqrt(ndotp * ndotp - pp * (nn - sphereRadius * sphereRadius))) / pp);
            }
        }
        return alpha;
    },
        1.0,
        thrust::minimum<HighP>());
}

template class BarrierEnergy<float>;
template class BarrierEnergy<double>;