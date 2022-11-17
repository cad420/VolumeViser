#pragma once

#include "Common.hpp"
#include <Model/Mesh.hpp>

class NeuronRenderer : public no_copy_t{
public:
    struct NeuronRendererCreateInfo{

    };
    explicit NeuronRenderer(const NeuronRendererCreateInfo& info);

    ~NeuronRenderer();

    void Reset();

    void AddNeuronMesh(const MeshData0& mesh, PatchID patch_id);

    /**
     * @brief 如果patch_id以及存在 那么会删除原来的数据
     */
    void AddNeuronMesh(const MeshData1& mesh, PatchID patch_id);

    void DeleteNeuronMesh(PatchID patch_id);

    void Draw(const mat4& view, const mat4& proj);

private:
    program_t shader;

    struct DrawPatch{
        mat4 model = mat4::identity();
        vertex_array_t vao;
        vertex_buffer_t<Vertex> vbo;
        index_buffer_t<uint32_t> ebo;
    };
    std::unordered_map<PatchID, DrawPatch> patches;

};