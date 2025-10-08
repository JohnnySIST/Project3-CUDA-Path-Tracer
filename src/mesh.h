#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <vector>

struct AABB {
  glm::vec3 low;
  glm::vec3 upper;
  AABB() : low(0, 0, 0), upper(0, 0, 0) {}
  AABB(glm::vec3 low, glm::vec3 upper) : low(low), upper(upper) {}
  AABB(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3);
  AABB(const AABB &a, const AABB &b);
};

struct BVHNode {
  BVHNode *left;
  BVHNode *right;
  AABB aabb;
  std::vector<int> triangles;
};

struct BVHNodeGPU {
    AABB aabb;
    int leftIndex;
    int rightIndex;
    int start;
    int end;
};

struct MeshGPU {
    glm::vec3* positions;
    glm::vec3* normals;
    int numTriangles;
    BVHNodeGPU* bvh_nodes;
    int* bvh_indices;
};

class Mesh {
public:
    Mesh() : materialid(-1), bvh(nullptr) {}
    Mesh(const std::string& filename) { loadFromGLTF(filename); }
    // ~Mesh() { freeBVH(bvh); }
    void loadFromGLTF(const std::string& filename);
    void buildBVH();
    void freeBVH(BVHNode* node);
    BVHNode* buildNode(std::vector<int> indices, int dim);
    bool comp(int t1, int t2, int dim) const;
    int flattenNodes(std::vector<BVHNodeGPU>& nodes, std::vector<int>& indices, BVHNode* node);
    void printBVH(BVHNode* node);

    int materialid;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;
    BVHNode* bvh;
};
