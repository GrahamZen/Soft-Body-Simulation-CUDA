CUDA-Accelerated Soft Body Simulation
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Gehan Zheng ([LinkedIn](https://www.linkedin.com/in/gehan-zheng-05877b24a/), [personal website](https://grahamzen.github.io/)),
Hanting Xu

## Click [here](https://github.com/GrahamZen/Soft-Body-Simulation-CUDA/tree/CIS5650-Final) for documentation (CIS5650 Final Project version)

## Requirements

- CUDA >= 12.0 (cublas, cusolver)
- CMake >= 3.18

## Description

This project is originally a final project for CIS5650 at UPenn. The goal of this toy project is to provide a CUDA-accelerated physical simulation framework with minimal dependencies. The framework is designed to be easily extensible, allowing new simulation algorithms, physical models, linear solvers, and collision detection methods to be added with minimal effort. The currently implemented features are listed below.

## Features

* Linear Solvers
    * [x] Sparse Cholesky Prefactorization W/ Approximate Minimum Degree Ordering
    * [x] Dense Cholesky Prefactorization
    * [x] Jacobi Solver (Naive)
    * [x] Cholesky Decomposition
    * [x] Preconditioned Conjugate Gradient

* FEM
    * [x] Projective Dynamics
      * [x] Direct (Cusolver's Cholesky)
      * [x] Chebyshev Acceleration
    * [x] Explicit Euler
    * [x] Incremental Potential Contact (IPC)
      * [x] Barrier
      * [ ] Frictional Contact
      * [ ] Dirichlet Boundary Conditions
        * [x] Equality Constraints
      * Materials
         * [x] Corotational
         * [x] Neo-Hookean

* Collision Detection
    * [x] Real-Time Bvh
    * [x] Ccd
    * [ ] Robust Collision Handling

## Dependencies

* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [CMake](https://cmake.org/download/)

Below are included in the project:

* OpenGL
* ImGui
* spdlog
* Eigen
* glfw
* catch2

## Screenshots

https://github.com/user-attachments/assets/1c088da8-6842-4ba1-9514-9ba6ddd2cf92

<p align="center">
<img src="image/showcase1.png" width="600">