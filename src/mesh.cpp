#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#include "mesh.h"

#include <iostream>

void Mesh::loadFromGLTF(const std::string& filename) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to load glTF: " << filename << std::endl;
    } else {
        std::cout << "Loaded glTF: " << filename << std::endl;
    }

    const tinygltf::Mesh& gltfMesh = model.meshes[0];
    for (const auto& primitive : gltfMesh.primitives) {
        if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            const float* positionsData = reinterpret_cast<const float*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
            printf("position count = %zu\n", accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                positions.emplace_back(positionsData[i * 3], positionsData[i * 3 + 1], positionsData[i * 3 + 2]);
            }
        }

        if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("NORMAL")];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            const float* normalsData = reinterpret_cast<const float*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
            printf("normal count = %zu\n", accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                normals.emplace_back(normalsData[i * 3], normalsData[i * 3 + 1], normalsData[i * 3 + 2]);
            }
        }

        if (primitive.indices >= 0) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            const unsigned short* indicesData = reinterpret_cast<const unsigned short*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
            printf("index count = %zu\n", accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                indices.emplace_back(indicesData[i]);
            }

            std::vector<glm::vec3> positions_temp;
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                positions_temp.push_back(positions[indices[i]]);
                positions_temp.push_back(positions[indices[i + 1]]);
                positions_temp.push_back(positions[indices[i + 2]]);
            }
            positions = positions_temp;
        }
    }
}

void Mesh::buildBVH() {
	std::vector<int> indices;
	for(int i = 0; i < positions.size()/3; i++){
		indices.push_back(i);
	}
	bvh = buildNode(indices, 0);
}

// build node on dimension dim, return built node
BVHNode* Mesh::buildNode(std::vector<int> vs, int dim) {
	BVHNode* r = new BVHNode();
	std::sort(vs.begin(), vs.end(), [&](int a, int b) { return comp(a, b, dim); });
	int mid = vs.size() / 2;
	if (mid < 2) { // only 1~3 triangles
		// r is leaf node
		r->triangles = std::move(vs);
		glm::vec3 v0, v1, v2;
		for (int i = 0; i < r->triangles.size(); i++) {
			int ind = r->triangles[i];
			v0 = positions[ind * 3];
			v1 = positions[ind * 3 + 1];
			v2 = positions[ind * 3 + 2];
			if (i == 0) r->aabb = AABB(v0, v1, v2);
			else r->aabb = AABB(r->aabb, AABB(v0, v1, v2));
		}
        r->left = nullptr;
        r->right = nullptr;
	} else {
		r->left = buildNode(std::vector<int>(vs.begin(), vs.begin() + mid), (dim + 1) % 3);
		r->right = buildNode(std::vector<int>(vs.begin() + mid, vs.end()), (dim + 1) % 3);
		r->aabb = AABB(r->left->aabb, r->right->aabb);
	}
	return r;
}

// compare two triangles on dimension dim(0,1,2)
bool Mesh::comp(int t1, int t2, int dim) const {
	glm::vec3 c1 = positions[t1 * 3] + positions[t1 * 3 + 1] + positions[t1 * 3 + 2];
	glm::vec3 c2 = positions[t2 * 3] + positions[t2 * 3 + 1] + positions[t2 * 3 + 2];
	return c1[dim] < c2[dim];
}

void Mesh::freeBVH(BVHNode* node) {
	if (node == nullptr) return;
	freeBVH(node->left);
	freeBVH(node->right);
	delete node;
}

AABB::AABB(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) {
  low = glm::min(v1, glm::min(v2, v3));
  upper = glm::max(v1, glm::max(v2, v3));
}

AABB::AABB(const AABB &a, const AABB &b) {
  low = glm::min(a.low, b.low);
  upper = glm::max(a.upper, b.upper);
}

bool AABB::intersect(glm::vec3 origin, glm::vec3 direction, float *t_in, float *t_out) {
    float dir_frac_x = (direction[0] == 0.0) ? 1.0e32 : 1.0 / direction[0];
    float dir_frac_y = (direction[1] == 0.0) ? 1.0e32 : 1.0 / direction[1];
    float dir_frac_z = (direction[2] == 0.0) ? 1.0e32 : 1.0 / direction[2];

    float tx1 = (low[0] - origin[0]) * dir_frac_x;
    float tx2 = (upper[0] - origin[0]) * dir_frac_x;
    float ty1 = (low[1] - origin[1]) * dir_frac_y;
    float ty2 = (upper[1] - origin[1]) * dir_frac_y;
    float tz1 = (low[2] - origin[2]) * dir_frac_z;
    float tz2 = (upper[2] - origin[2]) * dir_frac_z;

    *t_in = std::max(std::max(std::min(tx1, tx2), std::min(ty1, ty2)), std::min(tz1, tz2));
    *t_out = std::min(std::min(std::max(tx1, tx2), std::max(ty1, ty2)), std::max(tz1, tz2));

    if (*t_out < 0) return false;

    return *t_out >= *t_in;
}

int Mesh::flattenNodes(std::vector<BVHNodeGPU>& nodes, std::vector<int>& indices, BVHNode* node) {
    if (node == nullptr) return -1;
    int nodeIndex = nodes.size();
    nodes.emplace_back();
    BVHNodeGPU& nodeGPU = nodes.back();
    nodeGPU.aabb = node->aabb;
    if (node->left != nullptr) {
        nodeGPU.leftIndex = flattenNodes(nodes, indices, node->left);
    } else {
        nodeGPU.leftIndex = -1;
    }
    if (node->right != nullptr) {
        nodeGPU.rightIndex = flattenNodes(nodes, indices, node->right);
    } else {
        nodeGPU.rightIndex = -1;
    }
    if (node->left == nullptr || node->right == nullptr) {
        nodeGPU.start = indices.size();
        nodeGPU.end = indices.size() + node->triangles.size();
        for (int tri : node->triangles) {
            indices.push_back(tri);
        }
    } else {
        nodeGPU.start = nodes[nodeGPU.leftIndex].start;
        nodeGPU.end = nodes[nodeGPU.rightIndex].end;
    }
    return nodeIndex;
}

void Mesh::printBVH(BVHNode* node) {
    if (node == nullptr) return;
    std::cout << "Node " << node << " AABB: [(" << node->aabb.low.x << ", " << node->aabb.low.y << ", " << node->aabb.low.z << ") - ("
              << node->aabb.upper.x << ", " << node->aabb.upper.y << ", " << node->aabb.upper.z << ")]\n";
    if (node->left == nullptr && node->right == nullptr) {
        std::cout << "  Leaf with triangles: ";
        for (int tri : node->triangles) {
            std::cout << tri << " ";
        }
        std::cout << "\n";
    } else {
        printBVH(node->left);
        printBVH(node->right);
    }
}