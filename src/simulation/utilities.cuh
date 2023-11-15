#include <cuda.h>
#include <glm/glm.hpp>
#include <vector>
#include <GL/glew.h>
#include <utilities.h>

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line);

#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)
#define EPSILON 1e-6

template <typename T>
void inspectGLM(T* dev_ptr, int size) {
    std::vector<T> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
    inspectHost(host_ptr.data(), size);
}

template <typename T1, typename T2>
bool compareDevVSHost(T1* dev_ptr, T2* host_ptr2, int size) {
    std::vector<T1> host_ptr(size);
    cudaMemcpy(host_ptr.data(), dev_ptr, sizeof(T1) * size, cudaMemcpyDeviceToHost);
    return compareHostVSHost(host_ptr.data(), reinterpret_cast<T1*>(host_ptr2), size);
}

__inline__ __device__ float trace(const glm::mat3& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

__inline__ __device__ float trace2(const glm::mat3& a)
{
    return (float)((a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]));
}

__inline__ __device__ float trace4(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2]);
}

__inline__ __device__ float det2(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2]);
}



/*
__inline__ __device__ float rsqrt(float x) {
    // int ihalf = *(int *)&x - 0x00800000; // Alternative to next line,
    // float xhalf = *(float *)&ihalf;      // for sufficiently large nos.
    float xhalf = 0.5f * x;
    int i = *(int*)&x;          // View x as an int.
 // i = 0x5f3759df - (i >> 1);   // Initial guess (traditional).
    i = 0x5f375a82 - (i >> 1);   // Initial guess (slightly better).
    x = *(float*)&i;            // View i as float.
    x = x * (1.5f - xhalf * x * x);    // Newton step.
 // x = x*(1.5008908 - xhalf*x*x);  // Newton step for a balanced error.
    return x;
}*/

/* This is rsqrt with an additional step of the Newton iteration, for
increased accuracy. The constant 0x5f37599e makes the relative error
range from 0 to -0.00000463.
   You can't balance the error by adjusting the constant. */
__inline__ __device__ float rsqrt1(float x) {
    float xhalf = 0.5f * x;
    int i = *(int*)&x;          // View x as an int.
    i = 0x5f37599e - (i >> 1);   // Initial guess.
    x = *(float*)&i;            // View i as float.
    x = x * (1.5f - xhalf * x * x);    // Newton step.
    x = x * (1.5f - xhalf * x * x);    // Newton step again.
    return x;
}

__inline__ __device__ float accurateSqrt(float x)
{
    return x * rsqrt1(x);
}

__inline__ __device__ void condSwap(bool c, float& X, float& Y)
{
    // used in step 2
    float Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

__inline__ __device__ void condNegSwap(bool c, float& X, float& Y)
{
    // used in step 2 and 3
    float Z = -X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

// matrix multiplication M = A * B
__inline__ __device__ void multAB(float a11, float a12, float a13,
    float a21, float a22, float a23,
    float a31, float a32, float a33,
    //
    float b11, float b12, float b13,
    float b21, float b22, float b23,
    float b31, float b32, float b33,
    //
    float& m11, float& m12, float& m13,
    float& m21, float& m22, float& m23,
    float& m31, float& m32, float& m33)
{

    m11 = a11 * b11 + a12 * b21 + a13 * b31; m12 = a11 * b12 + a12 * b22 + a13 * b32; m13 = a11 * b13 + a12 * b23 + a13 * b33;
    m21 = a21 * b11 + a22 * b21 + a23 * b31; m22 = a21 * b12 + a22 * b22 + a23 * b32; m23 = a21 * b13 + a22 * b23 + a23 * b33;
    m31 = a31 * b11 + a32 * b21 + a33 * b31; m32 = a31 * b12 + a32 * b22 + a33 * b32; m33 = a31 * b13 + a32 * b23 + a33 * b33;
}

// matrix multiplication M = Transpose[A] * B
__inline__ __device__ void multAtB(float a11, float a12, float a13,
    float a21, float a22, float a23,
    float a31, float a32, float a33,
    //
    float b11, float b12, float b13,
    float b21, float b22, float b23,
    float b31, float b32, float b33,
    //
    float& m11, float& m12, float& m13,
    float& m21, float& m22, float& m23,
    float& m31, float& m32, float& m33)
{
    m11 = a11 * b11 + a21 * b21 + a31 * b31; m12 = a11 * b12 + a21 * b22 + a31 * b32; m13 = a11 * b13 + a21 * b23 + a31 * b33;
    m21 = a12 * b11 + a22 * b21 + a32 * b31; m22 = a12 * b12 + a22 * b22 + a32 * b32; m23 = a12 * b13 + a22 * b23 + a32 * b33;
    m31 = a13 * b11 + a23 * b21 + a33 * b31; m32 = a13 * b12 + a23 * b22 + a33 * b32; m33 = a13 * b13 + a23 * b23 + a33 * b33;
}

__inline__ __device__ void quatToMat3(const float* qV,
    float& m11, float& m12, float& m13,
    float& m21, float& m22, float& m23,
    float& m31, float& m32, float& m33
)
{
    float w = qV[3];
    float x = qV[0];
    float y = qV[1];
    float z = qV[2];

    float qxx = x * x;
    float qyy = y * y;
    float qzz = z * z;
    float qxz = x * z;
    float qxy = x * y;
    float qyz = y * z;
    float qwx = w * x;
    float qwy = w * y;
    float qwz = w * z;

    m11 = 1 - 2 * (qyy + qzz); m12 = 2 * (qxy - qwz); m13 = 2 * (qxz + qwy);
    m21 = 2 * (qxy + qwz); m22 = 1 - 2 * (qxx + qzz); m23 = 2 * (qyz - qwx);
    m31 = 2 * (qxz - qwy); m32 = 2 * (qyz + qwx); m33 = 1 - 2 * (qxx + qyy);
}

__inline__ __device__ void approximateGivensQuaternion(float a11, float a12, float a22, float& ch, float& sh)
{
    /*
         * Given givens angle computed by approximateGivensAngles,
         * compute the corresponding rotation quaternion.
         */
    ch = 2 * (a11 - a22);
    sh = a12;
    bool b = _gamma * sh * sh < ch* ch;
    // fast rsqrt function suffices
    // rsqrt2 (https://code.google.com/p/lppython/source/browse/algorithm/HDcode/newCode/rsqrt.c?r=26)
    // is even faster but results in too much error
    float w = rsqrt(ch * ch + sh * sh);
    ch = b ? w * ch : (float)_cstar;
    sh = b ? w * sh : (float)_sstar;
}

__inline__ __device__ void jacobiConjugation(const int x, const int y, const int z,
    float& s11,
    float& s21, float& s22,
    float& s31, float& s32, float& s33,
    float* qV)
{
    float ch, sh;
    approximateGivensQuaternion(s11, s21, s22, ch, sh);

    float scale = ch * ch + sh * sh;
    float a = (ch * ch - sh * sh) / scale;
    float b = (2 * sh * ch) / scale;

    // make temp copy of S
    float _s11 = s11;
    float _s21 = s21; float _s22 = s22;
    float _s31 = s31; float _s32 = s32; float _s33 = s33;

    // perform conjugation S = Q'*S*Q
    // Q already implicitly solved from a, b
    s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);	s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    s31 = a * _s31 + b * _s32;								s32 = -b * _s31 + a * _s32; s33 = _s33;

    // update cumulative rotation qV
    float tmp[3];
    tmp[0] = qV[0] * sh;
    tmp[1] = qV[1] * sh;
    tmp[2] = qV[2] * sh;
    sh *= qV[3];

    qV[0] *= ch;
    qV[1] *= ch;
    qV[2] *= ch;
    qV[3] *= ch;

    // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
    // for (p,q) = ((0,1),(1,2),(0,2))
    qV[z] += sh;
    qV[3] -= tmp[z]; // w
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    _s11 = s22;
    _s21 = s32; _s22 = s33;
    _s31 = s21; _s32 = s31; _s33 = s11;
    s11 = _s11;
    s21 = _s21; s22 = _s22;
    s31 = _s31; s32 = _s32; s33 = _s33;

}

__inline__ __device__ float dist2(float x, float y, float z)
{
    return x * x + y * y + z * z;
}

// finds transformation that diagonalizes a symmetric matrix
__inline__ __device__ void jacobiEigenanlysis( // symmetric matrix
    float& s11,
    float& s21, float& s22,
    float& s31, float& s32, float& s33,
    // quaternion representation of V
    float* qV)
{
    qV[3] = 1; qV[0] = 0; qV[1] = 0; qV[2] = 0; // follow same indexing convention as GLM
    for (int i = 0; i < 4; i++)
    {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
        //  asymptotic convergence
        jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
        jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
        jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
    }
}


__inline__ __device__ void sortSingularValues(// matrix that we want to decompose
    float& b11, float& b12, float& b13,
    float& b21, float& b22, float& b23,
    float& b31, float& b32, float& b33,
    // sort V simultaneously
    float& v11, float& v12, float& v13,
    float& v21, float& v22, float& v23,
    float& v31, float& v32, float& v33)
{
    float rho1 = dist2(b11, b21, b31);
    float rho2 = dist2(b12, b22, b32);
    float rho3 = dist2(b13, b23, b33);
    bool c;
    c = rho1 < rho2;
    condNegSwap(c, b11, b12); condNegSwap(c, v11, v12);
    condNegSwap(c, b21, b22); condNegSwap(c, v21, v22);
    condNegSwap(c, b31, b32); condNegSwap(c, v31, v32);
    condSwap(c, rho1, rho2);
    c = rho1 < rho3;
    condNegSwap(c, b11, b13); condNegSwap(c, v11, v13);
    condNegSwap(c, b21, b23); condNegSwap(c, v21, v23);
    condNegSwap(c, b31, b33); condNegSwap(c, v31, v33);
    condSwap(c, rho1, rho3);
    c = rho2 < rho3;
    condNegSwap(c, b12, b13); condNegSwap(c, v12, v13);
    condNegSwap(c, b22, b23); condNegSwap(c, v22, v23);
    condNegSwap(c, b32, b33); condNegSwap(c, v32, v33);
}


__inline__ __device__ void QRGivensQuaternion(float a1, float a2, float& ch, float& sh)
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    float epsilon = (float)EPSILON;
    float rho = accurateSqrt(a1 * a1 + a2 * a2);

    sh = rho > epsilon ? a2 : 0;
    ch = fabsf(a1) + fmaxf(rho, epsilon);
    bool b = a1 < 0;
    condSwap(b, sh, ch);
    float w = rsqrt(ch * ch + sh * sh);
    ch *= w;
    sh *= w;
}

__inline__ __device__ void QRDecomposition(// matrix that we want to decompose
    float b11, float b12, float b13,
    float b21, float b22, float b23,
    float b31, float b32, float b33,
    // output Q
    float& q11, float& q12, float& q13,
    float& q21, float& q22, float& q23,
    float& q31, float& q32, float& q33,
    // output R
    float& r11, float& r12, float& r13,
    float& r21, float& r22, float& r23,
    float& r31, float& r32, float& r33)
{
    float ch1, sh1, ch2, sh2, ch3, sh3;
    float a, b;

    // first givens rotation (ch,0,0,sh)
    QRGivensQuaternion(b11, b21, ch1, sh1);
    a = 1 - 2 * sh1 * sh1;
    b = 2 * ch1 * sh1;
    // apply B = Q' * B
    r11 = a * b11 + b * b21;  r12 = a * b12 + b * b22;  r13 = a * b13 + b * b23;
    r21 = -b * b11 + a * b21; r22 = -b * b12 + a * b22; r23 = -b * b13 + a * b23;
    r31 = b31;          r32 = b32;          r33 = b33;

    // second givens rotation (ch,0,-sh,0)
    QRGivensQuaternion(r11, r31, ch2, sh2);
    a = 1 - 2 * sh2 * sh2;
    b = 2 * ch2 * sh2;
    // apply B = Q' * B;
    b11 = a * r11 + b * r31;  b12 = a * r12 + b * r32;  b13 = a * r13 + b * r33;
    b21 = r21;           b22 = r22;           b23 = r23;
    b31 = -b * r11 + a * r31; b32 = -b * r12 + a * r32; b33 = -b * r13 + a * r33;

    // third givens rotation (ch,sh,0,0)
    QRGivensQuaternion(b22, b32, ch3, sh3);
    a = 1 - 2 * sh3 * sh3;
    b = 2 * ch3 * sh3;
    // R is now set to desired value
    r11 = b11;             r12 = b12;           r13 = b13;
    r21 = a * b21 + b * b31;     r22 = a * b22 + b * b32;   r23 = a * b23 + b * b33;
    r31 = -b * b21 + a * b31;    r32 = -b * b22 + a * b32;  r33 = -b * b23 + a * b33;

    // construct the cumulative rotation Q=Q1 * Q2 * Q3
    // the number of floating point operations for three quaternion multiplications
    // is more or less comparable to the explicit form of the joined matrix.
    // certainly more memory-efficient!
    float sh12 = sh1 * sh1;
    float sh22 = sh2 * sh2;
    float sh32 = sh3 * sh3;

    q11 = (-1 + 2 * sh12) * (-1 + 2 * sh22);
    q12 = 4 * ch2 * ch3 * (-1 + 2 * sh12) * sh2 * sh3 + 2 * ch1 * sh1 * (-1 + 2 * sh32);
    q13 = 4 * ch1 * ch3 * sh1 * sh3 - 2 * ch2 * (-1 + 2 * sh12) * sh2 * (-1 + 2 * sh32);

    q21 = 2 * ch1 * sh1 * (1 - 2 * sh22);
    q22 = -8 * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (-1 + 2 * sh12) * (-1 + 2 * sh32);
    q23 = -2 * ch3 * sh3 + 4 * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1 + 2 * sh32));

    q31 = 2 * ch2 * sh2;
    q32 = 2 * ch3 * (1 - 2 * sh22) * sh3;
    q33 = (-1 + 2 * sh22) * (-1 + 2 * sh32);
}

__device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const GLuint* Tet, int tet);
__device__ void svd(glm::mat3& A, glm::mat3& U, glm::mat3& S, glm::mat3& V);

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int number);
__global__ void AddGravity(glm::vec3* Force, glm::vec3* V, float mass, int numVerts, bool jump);
__global__ void computeInvDm(glm::mat3* inv_Dm, int tet_number, const glm::vec3* X, const GLuint* Tet);
__global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int tet_number, const GLuint* Tet);
__global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int number, const GLuint* Tet, float blendAlpha);
__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int tet_number);
__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* X, int number);
__global__ void ComputeForces(glm::vec3* Force, const glm::vec3* X, const GLuint* Tet, int tet_number, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1);
__global__ void UpdateParticles(glm::vec3* X, glm::vec3* V, const glm::vec3* Force,
    int number, float mass, float dt, float damp,
    glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN);

__global__ void HandleFloorCollision(glm::vec3* X, glm::vec3* V,
    int number, glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN);