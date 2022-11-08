#pragma once

#include "Common.hpp"

//对应一个SWC文件，一个SWC文件中可能有多个连通集，即多条神经元
class SWCRenderer final : public no_copy_t{
public:
    struct SWCRendererCreateInfo{
        int preserved_vertices_per_patch = 64;
        int preserved_indices_per_patch = 128;
    };

    using Vertex = vec4f;// pos:vec3 + radius:float

    explicit SWCRenderer(const SWCRendererCreateInfo& info);

    ~SWCRenderer();

    void Reset();

    void InitLine(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, size_t patch_id, const mat4& model = mat4::identity());

    //标注时不断添加新的点
    void AddLine(const Vertex& vertex_a, const Vertex& vertex_b, size_t patch_id);

    //对于标注线，如果中间的点删了，则需要直接删除整条线的数据然后重新添加该条线的所有数据
    void DeleteLine(size_t patch_id);

    void Draw(const mat4& view, const mat4& proj);


private:
    int preserved_vertices_per_patch;
    int preserved_indices_per_patch;
    //代表一条神经元，即连通的SWC点集形成的线
    struct DrawPatch{
        mat4 model;//todo: remove
        vertex_array_t vao;
        vertex_buffer_t<Vertex> vbo;
        index_buffer_t<uint32_t> ebo;
        int loaded_vertices_count;
        int loaded_indices_count;
    };
    struct Patch{
        std::unordered_map<Vertex, uint32_t> vertices_mp;
        DrawPatch draw_patch;
        uint32_t idx = 0;
    };
    std::unordered_map<size_t, Patch> patches;
    struct alignas(16) TransformParams{
        mat4 model;
        mat4 view;
        mat4 proj;
    }transform_params;
    std140_uniform_block_buffer_t<TransformParams> transform_params_buffer;
    program_t shader;
private:
    void ExpandIfFull(DrawPatch& patch);
};