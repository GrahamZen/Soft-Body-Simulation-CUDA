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

## Note on Configuration

The complete environment configuration is specified in context.json.

### Scene

The framework supports configuration of predefined soft bodies, rigid bodies, and camera parameters. Multiple contexts (scenes) can be loaded simultaneously, where each context may contain different combinations of soft and rigid objects, as well as distinct camera settings.

Each context can be configured independently with physical parameters such as time step size, gravity, damping coefficients, and friction coefficients, and supports real-time switching between contexts.

### Solver

The behavior of the solver can be adjusted by modifying parameters in each context. Currently, solvers supporting two floating-point precisions are available. When defining a context, setting the precision parameter to float uses the projective dynamics solver, while setting it to double uses the IPC solver. Only the parameters relevant to the active solver take effect.

The PD solver supports interactive object dragging within the scene. The IPC solver is significantly slower and consumes more GPU memory; therefore, it is not recommended for scenes involving objects with a large number of degrees of freedom. Different solvers expose different global solver and linear solver options in the ImGui combo box, which can be switched in real time. However, since solvers consume a substantial amount of GPU memory, frequent switching may lead to performance degradation. It is recommended to select the desired solver before starting the simulation and avoid switching after the simulation has begun.

## Screenshots

https://github.com/user-attachments/assets/1c088da8-6842-4ba1-9514-9ba6ddd2cf92

<p align="center">
<img src="image/showcase1.png" width="600">