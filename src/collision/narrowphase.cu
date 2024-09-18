#pragma once

#include <collision/bvh.h>
#include <collision/intersections.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>

struct CompareQuery {
    __host__ __device__
        bool operator()(const Query& q1, const Query& q2) const {
        if (q1.type != q2.type)
        {
            return q1.type < q2.type;
        }
        else
        {
            if (q1.type == QueryType::VF)
            {
                if (q1.v0 == q2.v0) {
                    return q1.toi < q2.toi;
                }
                return q1.v0 < q2.v0;
            }
            else
            {
                if (q1.v0 == q2.v0 && q1.v1 == q2.v1) {
                    return q1.toi < q2.toi;
                }
                else
                {
                    if (q1.v0 == q2.v0)
                        return q1.v1 < q2.v1;
                    else
                        return q1.v0 < q2.v0;
                }
            }
        }

    }
};

struct EqualQuery {
    __host__ __device__
        bool operator()(const Query& q1, const Query& q2) const {
        if (q1.type == q2.type)
        {
            if (q1.type == QueryType::VF)
            {
                return q1.v0 == q2.v0;
            }
            else // if is EE
            {
                return (q1.v0 == q2.v0 && q1.v1 == q2.v1);
            }
        }
        return false;
    }
};

template<typename Scalar>
__global__ void detectCollisionNarrow(int numQueries, Query* queries, const glm::tvec3<Scalar>* Xs, const glm::tvec3<Scalar>* XTildes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        glm::tvec3<Scalar> normal;
        Query& q = queries[index];
        q.toi = ccdCollisionTest(q, Xs, XTildes, normal);
        q.normal = normal;
    }
}

template<typename Scalar>
__global__ void storeTi(int numQueries, const Query* queries, Scalar* tI, glm::vec3* nors)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numQueries)
    {
        const Query& q = queries[index];

        if (q.type == QueryType::EE)
        {
            if (q.toi < 1.0f)
            {
                /*
                tI[q.v0] = q.toi;
                tI[q.v1] = q.toi;
                tI[q.v2] = q.toi;
                tI[q.v3] = q.toi;*/
                tI[q.v0] = 0.5f;
                tI[q.v1] = 0.5f;
                //tI[q.v2] = 0.5f;
                //tI[q.v3] = 0.5f;
                nors[q.v1] = q.normal;
                nors[q.v0] = q.normal;
            }
        }
        if (q.type == QueryType::VF)
        {
            if (q.toi < 1.0f)
            {
                tI[q.v0] = 0.5f;
                tI[q.v1] = 0.5f;
                tI[q.v2] = 0.5f;
                tI[q.v3] = 0.5f;
                nors[q.v0] = q.normal;
                nors[q.v1] = -q.normal;
                nors[q.v2] = -q.normal;
                nors[q.v3] = -q.normal;
            }
        }
        /*
        if (q.type == QueryType::VF)
        {
            tI[q.v0] = q.toi;
            nors[q.v0] = q.normal;
        }*/
    }
}

template<typename Scalar>
void CollisionDetection<Scalar>::NarrowPhase(const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, Scalar*& tI, glm::vec3*& nors)
{
    dim3 numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    detectCollisionNarrow << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries, X, XTilde);
    thrust::device_ptr<Query> dev_queriesPtr(dev_queries);

    thrust::sort(dev_queriesPtr, dev_queriesPtr + numQueries, CompareQuery());
    auto new_end = thrust::unique(dev_queriesPtr, dev_queriesPtr + numQueries, EqualQuery());
    numQueries = new_end - dev_queriesPtr;
    numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    storeTi << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries, tI, nors);
    cudaDeviceSynchronize();
}

struct getToi {
    __host__ __device__
        float operator()(const Query& q) const {
        return q.toi;
    }
};
template<typename Scalar>
Scalar CollisionDetection<Scalar>::NarrowPhase(const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde)
{
    dim3 numBlocksQuery = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    detectCollisionNarrow << <numBlocksQuery, threadsPerBlock >> > (numQueries, dev_queries, X, XTilde);
    thrust::device_ptr<Query> dev_queriesPtr(dev_queries);

    thrust::sort(dev_queriesPtr, dev_queriesPtr + numQueries, CompareQuery());
    auto new_end = thrust::unique(dev_queriesPtr, dev_queriesPtr + numQueries, EqualQuery());
    numQueries = new_end - dev_queriesPtr;
    return thrust::transform_reduce(dev_queriesPtr, dev_queriesPtr + numQueries, getToi(), 1.0f, thrust::minimum<Scalar>());
}

template class CollisionDetection<float>;
template class CollisionDetection<double>;