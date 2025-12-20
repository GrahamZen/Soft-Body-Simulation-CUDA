CUDA-Accelerated Soft Body Simulation
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Gehan Zheng ([LinkedIn](https://www.linkedin.com/in/gehan-zheng-05877b24a/), [personal website](https://grahamzen.github.io/)),
Hanting Xu

## Click [here](https://github.com/GrahamZen/Soft-Body-Simulation-CUDA/tree/CIS5650-Final) for documentation (CIS5650 Final Project version)

## Overview

This project is a CUDA-accelerated soft body simulation framework originally developed as a final project for **CIS 5650: GPU Programming and Architecture** at Upenn.

The goal of this project is to explore GPU-based physics simulation by building a **lightweight, extensible simulation framework** with minimal external dependencies. The system is designed to support rapid experimentation with different:

* physical models,
* numerical solvers,
* GPU-accelerated linear algebra pipelines.

---

## Features

* Linear Solvers
    * [x] Sparse Cholesky Prefactorization W/ Approximate Minimum Degree Ordering
    * [x] Dense Cholesky Prefactorization
    * [x] Jacobi Solver (Naive)
    * [x] Cholesky Decomposition
    * [x] Preconditioned Conjugate Gradient
      * [x] Incomplete Cholesky Preconditioner
      * [x] Jacobi Preconditioner

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
    * [x] Real-Time BVH Construction
    * [x] Continuous Collision Detection (CCD)

## Dependencies

### System Requirements

* **Operating System**

  * Windows
  * Linux
* **CUDA Toolkit** ≥ 12.0
  (cublas, cusolver required)
* **CMake** ≥ 3.18
* **OpenGL**

### Third-Party Libraries

The following libraries are included directly in the project:

* OpenGL
* ImGui
* GLFW
* Eigen
* spdlog
* Catch2

External tools:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
* [CMake](https://cmake.org/download/)

---

## Configuration

### Environment Configuration

The full runtime configuration is specified in `context.json`. This file defines simulation contexts, solver settings, and physical parameters.

---

### Scene Configuration

The framework supports multiple **simulation contexts**, each representing an independent scene. A context may contain:

* one or more soft bodies,
* rigid bodies,

Each context can be configured independently with physical parameters such as time step size, gravity, damping coefficients. Contexts can be switched **at runtime**.

---

### Solver Configuration

Solver behavior is controlled on a per-context basis.

* **Single-precision (`float`)**

  * Uses the **Projective Dynamics (PD)** solver
* **Double-precision (`double`)**

  * Uses the **Incremental Potential Contact (IPC)** solver

Only parameters relevant to the active solver are applied.

#### Notes on Solver Usage

* The PD solver supports **interactive object dragging**.
* IPC is **not recommended** for scenes with a large number of degrees of freedom; for large vertex counts, careful parameter tuning is required, otherwise the simulation may fail to converge and pause.
* For large-scale systems, **Cholesky-based solvers can become prohibitively slow**; **PCG with a Jacobi preconditioner** is recommended instead.
* Linear solvers can be switched via ImGui **before simulation starts**.

## Screenshots

https://github.com/user-attachments/assets/1c088da8-6842-4ba1-9514-9ba6ddd2cf92

<p align="center">
<img src="image/showcase1.png" width="600">