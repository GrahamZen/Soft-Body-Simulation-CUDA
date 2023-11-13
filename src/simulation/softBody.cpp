#include <softBody.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <glm/glm.hpp>

std::vector<GLuint> SoftBody::loadEleFile(const std::string& EleFilename, int startIndex)
{
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> tet_number;

    std::vector<GLuint> Tet(tet_number * 4);

    int a, b, c, d, e;
    for (int tet = 0; tet < tet_number && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Tet[tet * 4 + 0] = b - startIndex;
        Tet[tet * 4 + 1] = c - startIndex;
        Tet[tet * 4 + 2] = d - startIndex;
        Tet[tet * 4 + 3] = e - startIndex;
    }
    std::cout << "number of tetrahedrons: " << tet_number << std::endl;

    file.close();
    return Tet;
}

std::vector<glm::vec3> SoftBody::loadNodeFile(const std::string& nodeFilename, bool centralize) {
    std::ifstream file(nodeFilename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << nodeFilename << std::endl;
        return {};
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> number;
    std::vector<glm::vec3> X(number);
    glm::vec3 center(0.0f);

    for (int i = 0; i < number && std::getline(file, line); ++i) {
        std::istringstream lineStream(line);
        int index;
        float x, y, z;
        lineStream >> index >> x >> y >> z;

        X[i].x = x;
        X[i].y = y;
        X[i].z = z;

        center += X[i];
    }

    // Centralize the model
    if (centralize) {
        center /= static_cast<float>(number);
        for (int i = 0; i < number; ++i) {
            X[i] -= center;
            float temp = X[i].y;
            X[i].y = X[i].z;
            X[i].z = temp;
        }
    }
    std::cout << "number of nodes: " << number << std::endl;

    return X;
}