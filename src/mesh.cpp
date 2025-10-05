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