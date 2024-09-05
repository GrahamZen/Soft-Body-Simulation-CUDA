#include <energy/barrier.h>
#include <solver/solverUtil.cuh>
#include <fixedBodyData.h>
#include <plane.h>
#include <cylinder.h>
#include <sphere.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
    template <typename Scalar>
    __global__ void hessianKern(Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx, glm::tvec3<Scalar>* X, int numVerts,
        Plane* planes, int numPlanes, Cylinder* cylinders, int numCylinders, Sphere* spheres, int numSpheres,
        Scalar dhat, Scalar* contact_area, Scalar coef) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts)
            return;
        glm::tvec3<Scalar> x = X[idx];
        for (int j = 0; j < numPlanes; j++) {
            const Plane& plane = planes[j];
            glm::tvec3<Scalar> floorPos = glm::tvec3<Scalar>(plane.m_model[3]);
            glm::tvec3<Scalar> floorUp = plane.m_floorUp;
            Scalar d = glm::dot(x - floorPos, floorUp);
            if (d < dhat)
            {
                glm::tmat3x3<Scalar> hess = coef * contact_area[idx] * dhat * plane.kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(floorUp, floorUp);
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
            glm::tvec3<Scalar> axis = glm::tvec3<Scalar>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<Scalar> nnT = glm::tmat3x3<Scalar>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<Scalar> cylinderCenter = glm::tvec3<Scalar>(cy.m_model[3]);
            Scalar cylinderRadius = cy.m_radius;
            glm::tvec3<Scalar> n = nnT * (x - cylinderCenter);
            Scalar d = glm::length(n) - cylinderRadius;
            glm::tvec3<Scalar> normal = glm::normalize(n);
            if (d < dhat)
            {
                glm::tmat3x3<Scalar> hess = coef * contact_area[idx] * dhat * cy.kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(normal, normal);
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
            glm::tvec3<Scalar> sphereCenter = glm::tvec3<Scalar>(sphere.m_model[3]);
            Scalar sphereRadius = sphere.m_radius;
            glm::tvec3<Scalar> n = x - sphereCenter;
            Scalar d = glm::length(n) - sphereRadius;
            glm::tvec3<Scalar> normal = glm::normalize(n);
            if (d < dhat)
            {
                glm::tmat3x3<Scalar> hess = coef * contact_area[idx] * dhat * sphere.kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(normal, normal);
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

    template <typename Scalar>
    __global__ void gradientKern(Scalar* grad, glm::tvec3<Scalar>* X, int numVerts, Plane* planes, int numPlanes, Cylinder* cylinders, int numCylinders, Sphere* spheres, int numSpheres,
        Scalar dhat, Scalar* contact_area, Scalar coef) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts)
            return;
        const glm::tvec3<Scalar> x = X[idx];
        for (int j = 0; j < numPlanes; j++) {
            const Plane& plane = planes[j];
            glm::tvec3<Scalar> floorPos = glm::tvec3<Scalar>(plane.m_model[3]);
            glm::tvec3<Scalar> floorUp = plane.m_floorUp;
            Scalar d = glm::dot(x - floorPos, floorUp);
            if (d < dhat)
            {
                Scalar s = d / dhat;
                glm::tvec3<Scalar> gradient = coef * contact_area[idx] * dhat * (plane.kappa / 2 * (log(s) / dhat + (s - 1) / d)) * floorUp;
                grad[idx * 3] += gradient.x;
                grad[idx * 3 + 1] += gradient.y;
                grad[idx * 3 + 2] += gradient.z;
            }
        }
        for (int j = 0; j < numCylinders; j++) {
            const Cylinder cy = cylinders[j];
            glm::tvec3<Scalar> axis = glm::tvec3<Scalar>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<Scalar> nnT = glm::tmat3x3<Scalar>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<Scalar> cylinderCenter = glm::tvec3<Scalar>(cy.m_model[3]);
            Scalar cylinderRadius = cy.m_radius;
            glm::tvec3<Scalar> n = nnT * (x - cylinderCenter);
            Scalar d = glm::length(n) - cylinderRadius;
            glm::tvec3<Scalar> normal = glm::normalize(n);
            if (d < dhat)
            {
                Scalar s = d / dhat;
                glm::tvec3<Scalar> gradient = coef * contact_area[idx] * dhat * (cy.kappa / 2 * (log(s) / dhat + (s - 1) / d)) * normal;
                grad[idx * 3] += gradient.x;
                grad[idx * 3 + 1] += gradient.y;
                grad[idx * 3 + 2] += gradient.z;
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<Scalar> sphereCenter = glm::tvec3<Scalar>(sphere.m_model[3]);
            Scalar sphereRadius = sphere.m_radius;
            glm::tvec3<Scalar> n = x - sphereCenter;
            Scalar d = glm::length(n) - sphereRadius;
            glm::tvec3<Scalar> normal = glm::normalize(n);
            if (d < dhat)
            {
                Scalar s = d / dhat;
                glm::tvec3<Scalar> gradient = coef * contact_area[idx] * dhat * (sphere.kappa / 2 * (log(s) / dhat + (s - 1) / d)) * normal;
                grad[idx * 3] += gradient.x;
                grad[idx * 3 + 1] += gradient.y;
                grad[idx * 3 + 2] += gradient.z;
            }
        }
    }
}

template <typename Scalar>
int BarrierEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const { return solverData.numVerts * 9; }

template <typename Scalar>
BarrierEnergy<Scalar>::BarrierEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, Scalar dhat) :dhat(dhat), Energy<Scalar>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template <typename Scalar>
Scalar BarrierEnergy<Scalar>::Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData) const {
    const Plane* planes = solverData.pFixedBodies->dev_planes;
    const Cylinder* cylinders = solverData.pFixedBodies->dev_cylinders;
    const Sphere* spheres = solverData.pFixedBodies->dev_spheres;
    int numSpheres = solverData.pFixedBodies->numSpheres;
    int numCylinders = solverData.pFixedBodies->numCylinders;
    int numPlanes = solverData.pFixedBodies->numPlanes;
    Scalar dhat = this->dhat;
    Scalar sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=]__host__ __device__(indexType vertIdx) {
        const glm::tvec3<Scalar> x = Xs[vertIdx];
        Scalar sum = 0.0;
        for (int j = 0; j < numPlanes; j++)
        {
            const Plane& plane = planes[j];
            glm::tvec3<Scalar> floorPos = glm::tvec3<Scalar>(plane.m_model[3]);
            glm::tvec3<Scalar> floorUp = plane.m_floorUp;
            Scalar d = glm::dot(x - floorPos, floorUp);
            if (d < dhat)
            {
                Scalar s = d / dhat;
                sum += solverData.contact_area[vertIdx] * dhat * plane.kappa * 0.5 * (s - 1) * log(s);
            }
        }
        for (int j = 0; j < numCylinders; j++) {
            const Cylinder cy = cylinders[j];
            glm::tvec3<Scalar> axis = glm::tvec3<Scalar>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<Scalar> nnT = glm::tmat3x3<Scalar>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<Scalar> cylinderCenter = glm::tvec3<Scalar>(cy.m_model[3]);
            Scalar cylinderRadius = cy.m_radius;
            glm::tvec3<Scalar> n = nnT * (x - cylinderCenter);
            Scalar d = glm::length(n) - cylinderRadius;
            if (d < dhat)
            {
                Scalar s = d / dhat;
                sum += solverData.contact_area[vertIdx] * dhat * cy.kappa * 0.5 * (s - 1) * log(s);
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<Scalar> sphereCenter = glm::tvec3<Scalar>(sphere.m_model[3]);
            Scalar sphereRadius = sphere.m_radius;
            glm::tvec3<Scalar> n = x - sphereCenter;
            Scalar d = glm::length(n) - sphereRadius;
            glm::tvec3<Scalar> normal = glm::normalize(n);
            if (d < dhat)
            {
                Scalar s = d / dhat;
                sum += solverData.contact_area[vertIdx] * dhat * sphere.kappa * 0.5 * (s - 1) * log(s);
            }
        }
        return sum;
    },
        0.0,
        thrust::plus<Scalar>());
    return sum;
}

template<typename Scalar>
void BarrierEnergy<Scalar>::Gradient(Scalar* grad, const SolverData<Scalar>& solverData, Scalar coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::gradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.numVerts, solverData.pFixedBodies->dev_planes, solverData.pFixedBodies->numPlanes,
        solverData.pFixedBodies->dev_cylinders, solverData.pFixedBodies->numCylinders, solverData.pFixedBodies->dev_spheres, solverData.pFixedBodies->numSpheres, dhat, solverData.contact_area, coef);
}

template <typename Scalar>
void BarrierEnergy<Scalar>::Hessian(const SolverData<Scalar>& solverData, Scalar coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::hessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx, solverData.X, solverData.numVerts,
        solverData.pFixedBodies->dev_planes, solverData.pFixedBodies->numPlanes, solverData.pFixedBodies->dev_cylinders, solverData.pFixedBodies->numCylinders, solverData.pFixedBodies->dev_spheres, solverData.pFixedBodies->numSpheres, dhat, solverData.contact_area, coef);
}

template<typename Scalar>
Scalar BarrierEnergy<Scalar>::InitStepSize(const SolverData<Scalar>& solverData, Scalar* p) const
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
        Scalar alpha = 1.0;
        glm::tvec3<Scalar> localP{ -p[vertIdx * 3], -p[vertIdx * 3 + 1], -p[vertIdx * 3 + 2] };
        const glm::tvec3<Scalar> x = solverData.X[vertIdx];
        for (int j = 0; j < numPlanes; j++)
        {
            glm::tvec3<Scalar> floorUp = planes[j].m_floorUp;
            glm::tvec3<Scalar> floorPos = glm::tvec3<Scalar>(planes[j].m_model[3]);
            Scalar p_n = glm::dot(localP, floorUp);
            if (p_n < 0)
            {
                alpha = min(alpha, 0.9 * glm::dot(floorUp, x - floorPos) / -p_n);
            }
        }
        for (int j = 0; j < numCylinders; j++)
        {
            const Cylinder cy = cylinders[j];
            glm::tvec3<Scalar> axis = glm::tvec3<Scalar>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
            glm::tmat3x3<Scalar> nnT = glm::tmat3x3<Scalar>(1.f) - glm::outerProduct(axis, axis);
            glm::tvec3<Scalar> cylinderCenter = glm::tvec3<Scalar>(cy.m_model[3]);
            Scalar cylinderRadius = cy.m_radius;
            glm::tvec3<Scalar> n = nnT * (x - cylinderCenter);
            Scalar p_n = glm::dot(localP, n);
            if (p_n < 0)
            {
                glm::tvec3<Scalar> pp = nnT * localP;
                Scalar pp2 = glm::dot(pp, pp);
                Scalar n2 = glm::dot(n, n);
                Scalar ndotpp = glm::dot(n, pp);
                alpha = min(alpha, 0.9 * (-ndotpp - sqrt(ndotpp * ndotpp - pp2 * (n2 - cylinderRadius * cylinderRadius))) / pp2);
            }
        }
        for (int j = 0; j < numSpheres; j++) {
            const Sphere& sphere = spheres[j];
            glm::tvec3<Scalar> sphereCenter = glm::tvec3<Scalar>(sphere.m_model[3]);
            Scalar sphereRadius = sphere.m_radius;
            glm::tvec3<Scalar> n = x - sphereCenter;
            Scalar p_n = glm::dot(localP, n);
            if (p_n < 0)
            {
                Scalar ndotp = glm::dot(n, localP);
                Scalar pp = glm::dot(localP, localP);
                Scalar nn = glm::dot(n, n);
                alpha = min(alpha, 0.9 * (-ndotp - sqrt(ndotp * ndotp - pp * (nn - sphereRadius * sphereRadius))) / pp);
            }
        }
        return alpha;
    },
        1.0,
        thrust::minimum<Scalar>());
}

template class BarrierEnergy<float>;
template class BarrierEnergy<double>;