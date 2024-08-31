#pragma once

#include <iostream>
template<typename HighP, size_t N>
struct Vector {
    HighP value[N];

    __host__ __device__ Vector() {
        for (int i = 0; i < N; ++i) {
            value[i] = 0.0f;
        }
    }

    __host__ __device__ Vector(const glm::tmat3x3<HighP>& mat) {
        for (int i = 0; i < N; ++i) {
            value[i] = mat[i / 3][i % 3];
        }
    }

    __host__ __device__ Vector(HighP val) {
        for (int i = 0; i < N; ++i) {
            value[i] = val;
        }
    }

    __host__ __device__ Vector operator=(const Vector& vec) {
        for (int i = 0; i < N; ++i) {
            value[i] = vec[i];
        }
        return *this;
    }

    __host__ __device__ HighP& operator[](int index) {
        return value[index];
    }

    __host__ __device__ const HighP& operator[](int index) const {
        return value[index];
    }

    __host__ __device__ const HighP& operator*(const Vector& vec) const {
        HighP result = 0.0f;
        for (int i = 0; i < N; ++i) {
            result += value[i] * vec[i];
        }
        return result;
    }

    __host__ __device__ Vector operator*(HighP val) const {
        Vector result;
        for (int i = 0; i < N; ++i) {
            result[i] = value[i] * val;
        }
        return result;
    }

    __host__ __device__ Vector operator+(const Vector& vec) const {
        Vector result;
        for (int i = 0; i < N; ++i) {
            result[i] = value[i] + vec[i];
        }
        return result;
    }

    __host__ __device__ Vector operator-(const Vector& vec) const {
        Vector result;
        for (int i = 0; i < N; ++i) {
            result[i] = value[i] - vec[i];
        }
        return result;
    }

    __host__ __device__ Vector operator/(HighP val) const {
        Vector result;
        for (int i = 0; i < N; ++i) {
            result[i] = value[i] / val;
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& vec) {
        os << "[";
        for (int i = 0; i < N; ++i) {
            os << vec[i] << ", ";
        }
        os << "]";
        return os;
    }
};

template<typename HighP>
using Vector9 = Vector<HighP, 9>;
template<typename HighP>
using Vector12 = Vector<HighP, 12>;