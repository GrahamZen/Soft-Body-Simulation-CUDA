#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

#define PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON 0.00001f

class BVHNode;
class AABB;
class Query;
class Sphere;

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); // Thanks to http://stackoverflow.com/a/6089413
    glm::mat4 modelMatrix(const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale);
    template <typename T>
    void inspectHost(const T*, int);
    void inspectHost(const unsigned int*, int);
    void inspectHostMorton(const unsigned int* host_ptr, int size);
    void inspectHost(const BVHNode* hstBVHNodes, int size);
    void inspectHost(const AABB*, int);
    void inspectHost(const Query* query, int size);
    void inspectHost(const Sphere* spheres, int size);

    template <typename T>
    bool compareHostVSHost(const T* host_ptr1, const T* host_ptr2, int size);
    std::ifstream findFile(const std::string& fileName);
    std::string findFileName(const std::string& fileName);
}