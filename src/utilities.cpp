//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <cstdio>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <utilities.h>
#include <simulationContext.h>
#include <json.hpp>
#include <sceneStructs.h>

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

template <typename T>
void inspectHost(T* host_ptr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << glm::to_string(host_ptr[i]) << std::endl;
    }
}

template void inspectHost<glm::vec3>(glm::vec3* dev_ptr, int size);
template void inspectHost<glm::vec4>(glm::vec4* dev_ptr, int size);
template void inspectHost<glm::mat3>(glm::mat3* dev_ptr, int size);
template void inspectHost<glm::mat4>(glm::mat4* dev_ptr, int size);
void inspectHost(unsigned int* host_ptr, int size) {
    for (int i = 0; i < size / 4; i++) {
        std::cout << host_ptr[i * 4] << " " << host_ptr[i * 4 + 1] << " " << host_ptr[i * 4 + 2] << " " << host_ptr[i * 4 + 3] << std::endl;
    }
}

std::ifstream findFile(const std::string& fileName) {
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

SimulationCUDAContext* loadSimContext() {
    SimulationCUDAContext* simContext = new SimulationCUDAContext();

    std::string filename = "context.json";
    std::ifstream fileStream = findFile(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
        return nullptr;
    }
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();

    if (json.contains("dt")) {
        simContext->setDt(json["dt"].get<float>());
    }

    if (json.contains("softBodies")) {
        for (const auto& sbJson : json["softBodies"]) {
            std::string nodeFile = sbJson["nodeFile"];
            std::string eleFile = sbJson["eleFile"];
            glm::vec3 pos = glm::vec3(sbJson["pos"][0].get<float>(), sbJson["pos"][1].get<float>(), sbJson["pos"][2].get<float>());
            glm::vec3 scale = glm::vec3(sbJson["scale"][0].get<float>(), sbJson["scale"][1].get<float>(), sbJson["scale"][2].get<float>());
            glm::vec3 rot = glm::vec3(sbJson["rot"][0].get<float>(), sbJson["rot"][1].get<float>(), sbJson["rot"][2].get<float>());
            bool jump = sbJson["jump"].get<bool>();
            float mass = sbJson["mass"].get<float>();
            float stiffness_0 = sbJson["stiffness_0"].get<float>();
            float stiffness_1 = sbJson["stiffness_1"].get<float>();
            float damp = sbJson["damp"].get<float>();
            float muN = sbJson["muN"].get<float>();
            float muT = sbJson["muT"].get<float>();
            bool centralize = sbJson["centralize"].get<bool>();
            int startIndex = sbJson["start index"].get<int>();

            SoftBody* softBody = new SoftBody(nodeFile.c_str(), eleFile.c_str(), simContext,
                pos, scale, rot,
                mass, stiffness_0, stiffness_1, damp, muN, muT, centralize, startIndex);

            simContext->addSoftBody(softBody);
        }
    }

    return simContext;
}

Camera& computeCameraParams(Camera& camera)
{
    // assuming resolution, position, lookAt, view, up, fovy are already set
    float yscaled = tan(camera.fov.y * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov.x = fovx;

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);
    return camera;
}

bool loadContext(Camera& camera) {
    SimulationCUDAContext* simContext = new SimulationCUDAContext();

    std::string filename = "context.json";
    std::ifstream fileStream = findFile(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
        return false;
    }
    nlohmann::json json;
    fileStream >> json;
    fileStream.close();

    if (json.contains("camera")) {
        camera.resolution.y = json["camera"]["screen height"];
        float aspectRatio = json["camera"]["aspect ratio"];
        camera.resolution.x = aspectRatio * camera.resolution.y;
        camera.position = glm::vec3(json["camera"]["position"][0],
            json["camera"]["position"][1],
            json["camera"]["position"][2]);
        camera.lookAt = glm::vec3(json["camera"]["lookAt"][0],
            json["camera"]["lookAt"][1],
            json["camera"]["lookAt"][2]);
        camera.view = glm::vec3(json["camera"]["view"][0],
            json["camera"]["view"][1],
            json["camera"]["view"][2]);
        camera.up = glm::vec3(json["camera"]["up"][0],
            json["camera"]["up"][1],
            json["camera"]["up"][2]);
        camera.fov.y = json["camera"]["fovy"];
        computeCameraParams(camera);
    }

    return true;
}