#pragma once
#include <vector.h>
#include <iostream>

template<typename Scalar, size_t Rows, size_t Cols>
struct Matrix {
public:
    Vector<Scalar, Cols> value[Rows];

    __host__ __device__ Matrix() {
        for (int i = 0; i < Rows; ++i) {
            value[i] = Vector<Scalar, Cols>();
        }
    }

    __host__ __device__ Matrix(const Scalar& val) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                value[i][j] = (i == j) ? val : Scalar(0);
            }
        }
    }

    __host__ __device__ Matrix(const Vector<Scalar, Rows>& lhs, const Vector<Scalar, Cols>& rhs) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                value[i][j] = lhs[i] * rhs[j];
            }
        }
    }

    __host__ __device__ Vector<Scalar, Cols>& operator[](int index) {
        return value[index];
    }

    __host__ __device__ const Vector<Scalar, Cols>& operator[](int index) const {
        return value[index];
    }

    template<size_t M = Cols>
    __host__ __device__ Vector<Scalar, Rows> operator*(const Vector<Scalar, M>& vec) const {
        static_assert(M == Cols, "Matrix and vector dimensions do not match for multiplication");
        Vector<Scalar, Rows> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[i] += value[i][j] * vec[j];
            }
        }
        return result;
    }

    template<size_t N>
    __host__ __device__ Matrix<Scalar, Rows, N> operator*(const Matrix<Scalar, Cols, N>& mat) const {
        Matrix<Scalar, Rows, N> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < Cols; ++k) {
                    result[i][j] += value[i][k] * mat[k][j];
                }
            }
        }
        return result;
    }

    __host__ __device__ Matrix<Scalar, Cols, Rows> transpose() const {
        Matrix<Scalar, Cols, Rows> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[j][i] = value[i][j];
            }
        }
        return result;
    }


    __host__ __device__ Matrix operator*(const Scalar& val) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = value[i] * val;
        }
    }

    __host__ __device__ friend Matrix operator*(const Scalar& val, const Matrix& mat) {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = mat[i] * val;
        }
        return result;
    }

    __host__ __device__ Matrix operator+(const Matrix& mat) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = value[i] + mat[i];
        }
        return result;
    }

    __host__ __device__ Matrix operator-(const Matrix& mat) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = value[i] - mat[i];
        }
        return result;
    }

    __host__ __device__ Matrix operator/(const Scalar& val) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = value[i] / val;
        }
        return result;
    }

    __host__ __device__ Matrix operator+=(const Matrix& mat) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = value[i] + mat[i];
        }
        return *this;
    }

    __host__ __device__ Matrix operator-=(const Matrix& mat) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = value[i] - mat[i];
        }
        return *this;
    }

    __host__ __device__ Matrix operator*=(const Scalar& val) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = value[i] * val;
        }
        return *this;
    }

    __host__ __device__ Matrix operator/=(const Scalar& val) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = value[i] / val;
        }
        return *this;
    }

    template<size_t N>
    __host__ __device__ Matrix operator*=(const Matrix<Scalar, Cols, N>& mat) {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < Cols; ++k) {
                    result[i][j] += value[i][k] * mat[k][j];
                }
            }
        }
        return result;
    }

    __host__ __device__ Matrix operator=(const Matrix& mat) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = mat[i];
        }
        return *this;
    }

    __host__ __device__ Matrix operator=(const Scalar& val) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = Vector<Scalar, N>(val);
        }
        return *this;
    }

    __host__ __device__ Matrix operator-() const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = -value[i];
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
        for (int i = 0; i < Rows; ++i) {
            os << mat[i] << std::endl;
        }
        return os;
    }
};

template<typename Scalar>
using Matrix9 = Matrix<Scalar, 9, 9>;
template<typename Scalar>
using Matrix12 = Matrix<Scalar, 12, 12>;

template<typename Scalar>
using Matrix9x12 = Matrix<Scalar, 9, 12>;

template<typename Scalar>
using Matrix12x9 = Matrix<Scalar, 12, 9>;