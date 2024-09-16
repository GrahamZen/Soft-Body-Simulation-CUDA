#pragma once

#include <iostream>
#include <glm/glm.hpp>

template <typename Scalar, size_t N>
struct Vector;

template <typename Scalar>
struct VectorView
{
    size_t K;
    Scalar* value;

    __host__ __device__ Scalar& operator[](size_t i)
    {
        return value[i];
    }

    __host__ __device__ const Scalar& operator[](size_t i) const
    {
        return value[i];
    }

    template <size_t M>
    __host__ __device__ VectorView<Scalar>& operator=(const Vector<Scalar, M>& other)
    {
        for (size_t i = 0; i < K; ++i)
        {
            value[i] = other.value[i];
        }
        return *this;
    }

    __host__ __device__ VectorView<Scalar>& operator=(const VectorView<Scalar>& other) {
        for (size_t i = 0; i < K; ++i)
        {
            value[i] = other.value[i];
        }
        return *this;
    }

    template <typename OtherScalar>
    __host__ __device__ VectorView<Scalar>& operator=(const VectorView<const OtherScalar>& other)
    {
        for (size_t i = 0; i < K; ++i)
        {
            value[i] = other[i];
        }
        return *this;
    }
};

template <typename Scalar, size_t N>
struct Vector
{
public:
    __host__ __device__ VectorView<Scalar> head(size_t K)
    {
        return VectorView<Scalar>{K, value};
    }

    __host__ __device__ const VectorView<const Scalar> head(size_t K) const
    {
        return VectorView<const Scalar>{K, value};
    }

    __host__ __device__ VectorView<Scalar> tail(size_t K)
    {
        return VectorView<Scalar>{K, value + (N - K)};
    }

    __host__ __device__ const VectorView<const Scalar> tail(size_t K) const
    {
        return VectorView<const Scalar>{K, value + (N - K)};
    }

    __host__ __device__ VectorView<Scalar> segment(size_t K, size_t pos)
    {

        return VectorView<Scalar>{K, value + pos};
    }

    __host__ __device__ const VectorView<const Scalar> segment(size_t K, size_t pos) const
    {

        return VectorView<const Scalar>{K, value + pos};
    }
    Scalar value[N];
    __host__ __device__ Scalar* data() {
        return value;
    }

    __host__ __device__ Vector() {
        for (int i = 0; i < N; ++i) {
            value[i] = 0.0f;
        }
    }

    __host__ __device__ Vector(const glm::tmat3x3<Scalar>& mat) {
        for (int i = 0; i < N; ++i) {
            value[i] = mat[i / 3][i % 3];
        }
    }

    __host__ __device__ Vector(const glm::tvec3<Scalar>& vec) {
        for (int i = 0; i < N; ++i) {
            value[i] = vec[i];
        }
    }

    __host__ __device__ Vector(Scalar val) {
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

    __host__ __device__ Scalar& operator[](int index) {
        return value[index];
    }

    __host__ __device__ const Scalar& operator[](int index) const {
        return value[index];
    }

    __host__ __device__ const Scalar& operator*(const Vector& vec) const {
        Scalar result = 0.0f;
        for (int i = 0; i < N; ++i) {
            result += value[i] * vec[i];
        }
        return result;
    }

    __host__ __device__ Vector operator*(Scalar val) const {
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

    __host__ __device__ Vector operator/(Scalar val) const {
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

template<typename Scalar>
using Vector9 = Vector<Scalar, 9>;
template<typename Scalar>
using Vector12 = Vector<Scalar, 12>;

template<typename Scalar, size_t N>
__forceinline__ __host__ __device__ void printVectorFixed(const Vector<Scalar, N>& v, const char* name) {
    printf("Vector %d %s\n%f %f %f \n--------------------------------\n", N, name, v[0], v[1], v[2]);
}

template<typename Scalar, size_t N>
__forceinline__ __host__ __device__ void printVector(const Vector<Scalar, N>& v, const char* name) {
    printf("Vector %d %s\n", (int)N, name);
    for (size_t i = 0; i < N; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n--------------------------------\n");
}
