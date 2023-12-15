//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include <utilities.h>
#include <collision/aabb.h>
#include <sphere.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <bitset>
#include <type_traits>

template<typename T, typename = void>
struct has_glm_to_string : std::false_type {};

template<typename T>
struct has_glm_to_string<T, std::void_t<decltype(glm::to_string(std::declval<T>()))>> : std::true_type {};

template<typename T>
struct is_glm_type : has_glm_to_string<T> {};
namespace fs = std::filesystem;

float utilityCore::clamp(float f, float min, float max) {
    if (f < min) {
        return min;
    }
    else if (f > max) {
        return max;
    }
    else {
        return f;
    }
}

bool utilityCore::replaceString(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string utilityCore::convertIntToString(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

glm::vec3 utilityCore::clampRGB(glm::vec3 color) {
    if (color[0] < 0) {
        color[0] = 0;
    }
    else if (color[0] > 255) {
        color[0] = 255;
    }
    if (color[1] < 0) {
        color[1] = 0;
    }
    else if (color[1] > 255) {
        color[1] = 255;
    }
    if (color[2] < 0) {
        color[2] = 0;
    }
    else if (color[2] > 255) {
        color[2] = 255;
    }
    return color;
}

bool utilityCore::epsilonCheck(float a, float b) {
    if (fabs(fabs(a) - fabs(b)) < EPSILON) {
        return true;
    }
    else {
        return false;
    }
}

glm::mat4 utilityCore::buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

std::vector<std::string> utilityCore::tokenizeString(std::string str) {
    std::stringstream strstr(str);
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);
    return results;
}

std::istream& utilityCore::safeGetline(std::istream& is, std::string& t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if (t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}

glm::mat4 utilityCore::modelMatrix(const glm::vec3& pos, const glm::vec3& rot, const glm::vec3& scale) {
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, scale);
    return model;
}

template <typename T>
void utilityCore::inspectHost(const T* host_ptr, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    if constexpr (is_glm_type<T>::value) {
        for (int i = 0; i < size; i++) {
            std::cout << "glm::" << glm::to_string(host_ptr[i]) << "," << std::endl;
        }
    }
    else {
        for (int i = 0; i < size; i++) {
            std::cout << host_ptr[i] << std::endl;
        }
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}

template void utilityCore::inspectHost<glm::vec3>(const glm::vec3* dev_ptr, int size);
template void utilityCore::inspectHost<glm::vec4>(const glm::vec4* dev_ptr, int size);
template void utilityCore::inspectHost<glm::mat3>(const glm::mat3* dev_ptr, int size);
template void utilityCore::inspectHost<glm::mat4>(const glm::mat4* dev_ptr, int size);
template void utilityCore::inspectHost<int>(const int*, int);
template void utilityCore::inspectHost<dataType>(const dataType*, int);

void utilityCore::inspectHost(const BVHNode* hstBVHNodes, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << i << ": " << hstBVHNodes[i].leftIndex << "," << hstBVHNodes[i].rightIndex << "  parent:" << hstBVHNodes[i].parent << std::endl;
        std::cout << i << ": " << hstBVHNodes[i].bbox.max.x << "," << hstBVHNodes[i].bbox.max.y << "," << hstBVHNodes[i].bbox.max.z << std::endl;
        //cout << i << ": " << hstBVHNodes[i].bbox.min.x << "," << hstBVHNodes[i].bbox.min.y << "," << hstBVHNodes[i].bbox.min.z << endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}

void utilityCore::inspectHost(const AABB* aabb, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << "min: " << glm::to_string(aabb[i].min) << ", max:" << glm::to_string(aabb[i].max) << std::endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}

void utilityCore::inspectHost(const Query* query, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    for (int i = 0; i < size; i++) {
        if (query[i].type == QueryType::EE)
            std::cout << "EE:";
        if (query[i].type == QueryType::VF)
            std::cout << "VF:";
        if (query[i].type == QueryType::UNKNOWN)
            std::cout << "UNKNOWN:";
        std::cout << query[i].v0 << "," << query[i].v1 << "," << query[i].v2 << "," << query[i].v3 << ", t:" << query[i].toi
            << ", n:" << glm::to_string(query[i].normal) << std::endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}

void utilityCore::inspectHost(const Sphere* spheres, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << spheres[i] << std::endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}


template <typename T>
bool utilityCore::compareHostVSHost(const T* host_ptr1, const T* host_ptr2, int size) {
    for (int i = 0; i < size; i++) {
        if (host_ptr1[i] != host_ptr2[i]) {
            std::cout << "Failed:" << std::endl
                << "host_ptr1[" << i << "] = " << glm::to_string(host_ptr1[i]) << ", "
                << "host_ptr2[" << i << "] = " << glm::to_string(host_ptr2[i]) << std::endl;
            return false;
        }
    }
    return true;
}
template bool utilityCore::compareHostVSHost<glm::vec3>(const glm::vec3*, const glm::vec3*, int size);

void utilityCore::inspectHost(const unsigned int* host_ptr, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    for (int i = 0; i < size / 4; i++) {
        std::cout << host_ptr[i * 4] << " " << host_ptr[i * 4 + 1] << " " << host_ptr[i * 4 + 2] << " " << host_ptr[i * 4 + 3] << std::endl;
    }
    int remain = size % 4;
    if (remain != 0) {
        for (int i = size - remain; i < size; i++) {
            std::cout << host_ptr[i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}

void utilityCore::inspectHostMorton(const unsigned int* host_ptr, int size) {
    std::cout << "---------------------------inspectHost--------------------------------" << std::endl;
    for (int i = 0; i < 20; i++)
    {
        std::cout << std::bitset<30>(host_ptr[i]) << std::endl;
    }
    std::cout << "------------------------inspectHost--END------------------------------" << std::endl;
}

std::ifstream utilityCore::findFile(const std::string& fileName) {
    fs::path currentPath = fs::current_path();
    for (int i = 0; i < 5; ++i) {
        fs::path filePath = currentPath / fileName;
        if (fs::exists(filePath)) {
            std::ifstream fileStream(filePath);
            if (fileStream.is_open())
                return fileStream;
        }
        currentPath = currentPath.parent_path();
    }

    std::cerr << "File not found: " << fileName << std::endl;
    return std::ifstream();
}

std::string utilityCore::findFileName(const std::string& fileName) {
    fs::path currentPath = fs::current_path();
    for (int i = 0; i < 5; ++i) {
        fs::path filePath = currentPath / fileName;
        if (fs::exists(filePath)) {
            return filePath.string();
        }
        currentPath = currentPath.parent_path();
    }

    std::cerr << "File not found: " << fileName << std::endl;
    return std::string();
}