enum class PointTriangleDistanceType {
    P_T0, ///< The point is closest to triangle vertex zero.
    P_T1, ///< The point is closest to triangle vertex one.
    P_T2, ///< The point is closest to triangle vertex two.
    P_E0, ///< The point is closest to triangle edge zero (vertex zero to one).
    P_E1, ///< The point is closest to triangle edge one (vertex one to two).
    P_E2, ///< The point is closest to triangle edge two (vertex two to zero).
    P_T,  ///< The point is closest to the interior of the triangle.
    AUTO  ///< Automatically determine the closest pair.
};


__host__ __device__ void ldltDecomposition(const glm::mat3& A, glm::mat3& L, glm::mat3& D) {
    L = glm::mat3(1.0);
    D = glm::mat3(0.0);

    for (int i = 0; i < 3; i++) {
        float sum = 0.0f;
        
        for (int j = 0; j < i; j++) {
            float lsum = 0.0f;
            for (int k = 0; k < j; k++) {
                lsum += L[i][k] * L[j][k] * D[k][k];
            }
            L[i][j] = (A[i][j] - lsum) / D[j][j];
            sum += L[i][j] * L[i][j] * D[j][j];
        }

        D[i][i] = A[i][i] - sum;
    }
}

__host__ __device__ glm::vec3 forwardSubstitution(const glm::mat3& L, const glm::vec3& b) {
    glm::vec3 y(0.0f);
    for (int i = 0; i < 3; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    return y;
}

__host__ __device__ glm::vec3 diagonalSolve(const glm::mat3& D, const glm::vec3& y) {
    glm::vec3 z(0.0f);
    for (int i = 0; i < 3; ++i) {
        z[i] = y[i] / D[i][i];
    }
    return z;
}

__host__ __device__ glm::vec3 backwardSubstitution(const glm::mat3& L, const glm::vec3& z) {
    glm::vec3 x(0.0f);
    for (int i = 2; i >= 0; --i) {
        float sum = 0.0f;
        for (int j = i + 1; j < 3; ++j) {
            sum += L[j][i] * x[j];
        }
        x[i] = (z[i] - sum) / L[i][i];
    }
    return x;
}

__host__ __device__ glm::vec3 matrixSolve(const glm::mat3& L, const glm::mat3& D, const glm::vec3& b) {
    glm::vec3 y = forwardSubstitution(L, b);
    glm::vec3 z = diagonalSolve(D, y);
    glm::vec3 x = backwardSubstitution(L, z);
    return x;
}

__host__ __device__ glm::vec2 matrixSolve(const glm::mat2x3 &basis, const glm::vec3 &rhs) {
    glm::mat3 ldltL, ldltD;
    ldltDecomposition(glm::transpose(basis) * basis, ldltL, ldltD);
    glm::vec3 x = matrixSolve(ldltL, ldltD, glm::transpose(basis) * rhs);
    return glm::vec2(x);
}

__host__ __device__ PointTriangleDistanceType point_triangle_distance_type(
    const glm::vec3& p,
    const glm::vec3& t0,
    const glm::vec3& t1,
    const glm::vec3& t2)
{
    glm::vec3 normal = glm::cross(t1 - t0, t2 - t0);

    glm::mat2x3 basis;
    glm::vec2 param[3];

    // Edge 0
    basis[0] = t1 - t0;
    basis[1] = glm::cross(basis[0], normal);
    param[0] = matrixSolve(basis, p - t0);
    if (param[0].x > 0.0 && param[0].x < 1.0 && param[0].y >= 0.0) {
        return PointTriangleDistanceType::P_E0;
    }

    // Edge 1
    basis[0] = t2 - t1;
    basis[1] = glm::cross(basis[0], normal);
    param[1] = matrixSolve(basis, p - t1);
    if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
        return PointTriangleDistanceType::P_E1;
    }

    // Edge 2
    basis[0] = t0 - t2;
    basis[1] = glm::cross(basis[0], normal);
    param[2] = matrixSolve(basis, p - t2);
    if (param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0) {
        return PointTriangleDistanceType::P_E2;
    }

    // Checking vertices
    if (param[0].x <= 0.0 && param[2].x >= 1.0) {
        return PointTriangleDistanceType::P_T0;
    } else if (param[1].x <= 0.0 && param[0].x >= 1.0) {
        return PointTriangleDistanceType::P_T1;
    } else if (param[2].x <= 0.0 && param[1].x >= 1.0) {
        return PointTriangleDistanceType::P_T2;
    }

    return PointTriangleDistanceType::P_T;
}
