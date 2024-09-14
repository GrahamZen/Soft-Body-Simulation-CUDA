﻿#pragma once
#include <aabb.h>
#include <distance/point_triangle.h>
#include <distance/edge_edge.h>
#include <cuda_runtime.h>

template <typename Scalar>
__global__ void GetDistanceType(const glm::tvec3<Scalar>* Xs, Query* queries, int numQueries);
template <typename Scalar>
__global__ void ComputeDistance(const glm::tvec3<Scalar>* Xs, Query* queries, int numQueries, Scalar dhat = 1e-3);

template<typename Scalar>
__device__ DistanceType point_triangle_distance_type(
    const glm::tvec3<Scalar>& p,
    const glm::tvec3<Scalar>& t0,
    const glm::tvec3<Scalar>& t1,
    const glm::tvec3<Scalar>& t2);

template<typename Scalar>
__device__ DistanceType edge_edge_distance_type(
    const glm::tvec3<Scalar>& ea0,
    const glm::tvec3<Scalar>& ea1,
    const glm::tvec3<Scalar>& eb0,
    const glm::tvec3<Scalar>& eb1);