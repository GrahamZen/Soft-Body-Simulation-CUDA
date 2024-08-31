#pragma once
#include <vector.h>
#include <iostream>

template<typename HighP, size_t Rows, size_t Cols>
struct Matrix {
public:
    Vector<HighP, Cols> value[Rows];

    __host__ __device__ Matrix() {
        for (int i = 0; i < Rows; ++i) {
            value[i] = Vector<HighP, Cols>();
        }
    }

    __host__ __device__ Matrix(const HighP& val) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                value[i][j] = (i == j) ? val : HighP(0);
            }
        }
    }

    __host__ __device__ Matrix(const Vector<HighP, Rows>& lhs, const Vector<HighP, Cols>& rhs) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                value[i][j] = lhs[i] * rhs[j];
            }
        }
    }

    __host__ __device__ Vector<HighP, Cols>& operator[](int index) {
        return value[index];
    }

    __host__ __device__ const Vector<HighP, Cols>& operator[](int index) const {
        return value[index];
    }

    template<size_t M = Cols>
    __host__ __device__ Vector<HighP, Rows> operator*(const Vector<HighP, M>& vec) const {
        static_assert(M == Cols, "Matrix and vector dimensions do not match for multiplication");
        Vector<HighP, Rows> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[i] += value[i][j] * vec[j];
            }
        }
        return result;
    }

    template<size_t N>
    __host__ __device__ Matrix<HighP, Rows, N> operator*(const Matrix<HighP, Cols, N>& mat) const {
        Matrix<HighP, Rows, N> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < Cols; ++k) {
                    result[i][j] += value[i][k] * mat[k][j];
                }
            }
        }
        return result;
    }

    __host__ __device__ Matrix<HighP, Cols, Rows> transpose() const {
        Matrix<HighP, Cols, Rows> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[j][i] = value[i][j];
            }
        }
        return result;
    }


    __host__ __device__ Matrix operator*(const HighP& val) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = value[i] * val;
        }
    }

    __host__ __device__ friend Matrix operator*(const HighP& val, const Matrix& mat) {
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

    __host__ __device__ Matrix operator/(const HighP& val) const {
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

    __host__ __device__ Matrix operator*=(const HighP& val) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = value[i] * val;
        }
        return *this;
    }

    __host__ __device__ Matrix operator/=(const HighP& val) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = value[i] / val;
        }
        return *this;
    }

    template<size_t N>
    __host__ __device__ Matrix operator*=(const Matrix<HighP, Cols, N>& mat) {
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

    __host__ __device__ Matrix operator=(const HighP& val) {
        for (int i = 0; i < Rows; ++i) {
            value[i] = Vector<HighP, N>(val);
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

template<typename HighP>
using Matrix9 = Matrix<HighP, 9, 9>;
template<typename HighP>
using Matrix12 = Matrix<HighP, 12, 12>;

template<typename HighP>
using Matrix9x12 = Matrix<HighP, 9, 12>;

template<typename HighP>
using Matrix12x9 = Matrix<HighP, 12, 9>;