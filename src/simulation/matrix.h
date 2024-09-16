#pragma once
#include <vector.h>
#include <iostream>
#include <cassert>

template<typename Scalar, size_t Rows, size_t Cols>
struct Matrix;
template <typename Scalar, size_t Rows, size_t Cols>
struct Block {
    Matrix<Scalar, Rows, Cols>& matrix;
    size_t rowOffset, colOffset;
    size_t blockRows, blockCols;
public:
    Block(Matrix<Scalar, Rows, Cols>& mat, size_t rowOff, size_t colOff, size_t blockR, size_t blockC)
        : matrix(mat), rowOffset(rowOff), colOffset(colOff), blockRows(blockR), blockCols(blockC) {}

    Scalar& operator()(size_t i, size_t j) {
        assert(i < blockRows && j < blockCols);
        return matrix[rowOffset + i][colOffset + j];
    }

    const Scalar& operator()(size_t i, size_t j) const {
        assert(i < blockRows && j < blockCols);
        return matrix[rowOffset + i][colOffset + j];
    }

    template <size_t OtherRows, size_t OtherCols>
    Block& operator=(const Block<Scalar, OtherRows, OtherCols>& other) {
        assert(blockRows == other.blockRows && blockCols == other.blockCols);
        for (size_t i = 0; i < blockRows; ++i) {
            for (size_t j = 0; j < blockCols; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
        return *this;
    }
    template <size_t OtherRows, size_t OtherCols>
    Block& operator=(const Matrix<Scalar, OtherRows, OtherCols>& other) {
        for (size_t i = 0; i < blockRows; ++i) {
            for (size_t j = 0; j < blockCols; ++j) {
                (*this)(i, j) = other[i][j];
            }
        }
        return *this;
    }
};

template<typename Scalar, size_t Rows, size_t Cols>
struct Matrix {
    Vector<Scalar, Cols> value[Rows];

public:
    __host__ __device__ Scalar* data() {
        return value[0].data();
    }

    Block<Scalar, Rows, Cols> block(size_t blockRows, size_t blockCols, size_t row, size_t col) {
        return Block<Scalar, Rows, Cols>(*this, row, col, blockRows, blockCols);
    }

    const Block<Scalar, Rows, Cols> block(size_t row, size_t col, size_t blockRows, size_t blockCols) const {
        return Block<Scalar, Rows, Cols>(*this, row, col, blockRows, blockCols);
    }

    Block<Scalar, Rows, Cols> topLeftCorner(size_t blockRows, size_t blockCols) {
        return block(blockRows, blockCols, 0, 0);
    }

    const Block<Scalar, Rows, Cols> topLeftCorner(size_t blockRows, size_t blockCols) const {
        return block(blockRows, blockCols, 0, 0);
    }

    Block<Scalar, Rows, Cols> topRightCorner(size_t blockRows, size_t blockCols) {
        return block(blockRows, blockCols, 0, Cols - blockCols);
    }

    const Block<Scalar, Rows, Cols> topRightCorner(size_t blockRows, size_t blockCols) const {
        return block(blockRows, blockCols, 0, Cols - blockCols);
    }

    Block<Scalar, Rows, Cols> bottomLeftCorner(size_t blockRows, size_t blockCols) {
        return block(blockRows, blockCols, Rows - blockRows, 0);
    }

    const Block<Scalar, Rows, Cols> bottomLeftCorner(size_t blockRows, size_t blockCols) const {
        return block(blockRows, blockCols, Rows - blockRows, 0);
    }

    Block<Scalar, Rows, Cols> bottomRightCorner(size_t blockRows, size_t blockCols) {
        return block(blockRows, blockCols, Rows - blockRows, Cols - blockCols);
    }

    const Block<Scalar, Rows, Cols> bottomRightCorner(size_t blockRows, size_t blockCols) const {
        return block(blockRows, blockCols, Rows - blockRows, Cols - blockCols);
    }

    Matrix(std::initializer_list<Scalar> list) {
        auto it = list.begin();
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                if (it != list.end()) {
                    value[i][j] = *it;
                    ++it;
                }
                else {
                }
            }
        }
    }

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
            value[i] = Vector<Scalar, Cols>(val);
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
using Matrix3 = Matrix<Scalar, 3, 3>;
template<typename Scalar>
using Matrix6 = Matrix<Scalar, 6, 6>;
template<typename Scalar>
using Matrix9 = Matrix<Scalar, 9, 9>;
template<typename Scalar>
using Matrix12 = Matrix<Scalar, 12, 12>;

template<typename Scalar>
using Matrix9x12 = Matrix<Scalar, 9, 12>;

template<typename Scalar>
using Matrix12x9 = Matrix<Scalar, 12, 9>;


template<typename Scalar, size_t Rows, size_t Cols>
__forceinline__ __host__ __device__ void printMatrixFixed(const Matrix<Scalar, Rows, Cols>& m, const char* name) {
    printf("Matrix %d x %d %s\n%f %f %f\n%f %f %f\n%f %f %f\n--------------------------------\n", Rows, Cols, name,
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2]);
}

template<typename Scalar, size_t Rows, size_t Cols>
__forceinline__ __host__ __device__ void printMatrix(const Matrix<Scalar, Rows, Cols>& m, const char* name) {
    printf("Matrix %d x %d %s\n--------------------------------\n", Rows, Cols, name);
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
}

template<typename Scalar>
__forceinline__ __host__ __device__ void printGLMMatrix(const glm::tmat3x3<Scalar>& m, const char* name) {
    printf("Matrix 3 x 3 %s\n%f %f %f\n%f %f %f\n%f %f %f\n--------------------------------\n", name,
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2]);
}


template <typename Scalar>
__device__ Matrix9x12<Scalar> ComputePFPx(const glm::tmat3x3<Scalar>& DmInv)
{
    const Scalar m = DmInv[0][0];
    const Scalar n = DmInv[1][0];
    const Scalar o = DmInv[2][0];
    const Scalar p = DmInv[0][1];
    const Scalar q = DmInv[1][1];
    const Scalar r = DmInv[2][1];
    const Scalar s = DmInv[0][2];
    const Scalar t = DmInv[1][2];
    const Scalar u = DmInv[2][2];
    const Scalar t1 = -m - p - s;
    const Scalar t2 = -n - q - t;
    const Scalar t3 = -o - r - u;
    Matrix9x12<Scalar> PFPx;
    PFPx[0][0] = t1;
    PFPx[0][3] = m;
    PFPx[0][6] = p;
    PFPx[0][9] = s;
    PFPx[1][1] = t1;
    PFPx[1][4] = m;
    PFPx[1][7] = p;
    PFPx[1][10] = s;
    PFPx[2][2] = t1;
    PFPx[2][5] = m;
    PFPx[2][8] = p;
    PFPx[2][11] = s;
    PFPx[3][0] = t2;
    PFPx[3][3] = n;
    PFPx[3][6] = q;
    PFPx[3][9] = t;
    PFPx[4][1] = t2;
    PFPx[4][4] = n;
    PFPx[4][7] = q;
    PFPx[4][10] = t;
    PFPx[5][2] = t2;
    PFPx[5][5] = n;
    PFPx[5][8] = q;
    PFPx[5][11] = t;
    PFPx[6][0] = t3;
    PFPx[6][3] = o;
    PFPx[6][6] = r;
    PFPx[6][9] = u;
    PFPx[7][1] = t3;
    PFPx[7][4] = o;
    PFPx[7][7] = r;
    PFPx[7][10] = u;
    PFPx[8][2] = t3;
    PFPx[8][5] = o;
    PFPx[8][8] = r;
    PFPx[8][11] = u;
    return PFPx;
}