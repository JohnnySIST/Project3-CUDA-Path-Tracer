#pragma once

#include <glm/glm.hpp>
#include <vector>

class Mesh {
public:
    Mesh() : materialid(-1) {}
    Mesh(const std::string& filename) { loadFromGLTF(filename); }
    void loadFromGLTF(const std::string& filename);

    int materialid;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;
};

struct MeshGPU {
    glm::vec3* positions;
    glm::vec3* normals;
    int numTriangles;
};