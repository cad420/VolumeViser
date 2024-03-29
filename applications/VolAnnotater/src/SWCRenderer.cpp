#include "SWCRenderer.hpp"

SWCRenderer::SWCRenderer(const SWCRenderer::SWCRendererCreateInfo &info) {
    shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("./glsl/line_shading.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("./glsl/line_shading.frag")
            );

    pt_shader = program_t::build_from(
        shader_t<GL_VERTEX_SHADER>::from_file("./glsl/pt_shading.vert"),
        shader_t<GL_FRAGMENT_SHADER>::from_file("./glsl/pt_shading.frag"),
        shader_t<GL_GEOMETRY_SHADER>::from_file("./glsl/pt_shading.geom")
        );

    preserved_vertices_per_patch = info.preserved_vertices_per_patch;
    preserved_indices_per_patch = info.preserved_indices_per_patch;

    transform_params_buffer.initialize_handle();
    transform_params_buffer.reinitialize_buffer_data(nullptr, GL_DYNAMIC_DRAW);
}

SWCRenderer::~SWCRenderer() {

}

void SWCRenderer::Reset() {
    patches.clear();

}

void SWCRenderer::AddVertex(const std::vector<Vertex> &vertices)
{
    sel_tags.vao.destroy();
    sel_tags.vbo.destroy();

    sel_tags.loaded_vertices_count = vertices.size();
    if(sel_tags.loaded_vertices_count == 0) return;
    sel_tags.vao.initialize_handle();
    sel_tags.vbo.initialize_handle();
    sel_tags.vbo.reinitialize_buffer_data(nullptr, vertices.size(), GL_DYNAMIC_DRAW);
    sel_tags.vao.bind_vertex_buffer_to_attrib(attrib_var_t<Vertex>(0), sel_tags.vbo, 0);
    sel_tags.vao.enable_attrib(attrib_var_t<Vertex>(0));
    sel_tags.vbo.set_buffer_data(vertices.data(), 0, vertices.size());
}

void SWCRenderer::ClearVertex()
{
    sel_tags.loaded_vertices_count = 0;
}

void SWCRenderer::InitLine(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, size_t patch_id, const mat4& model) {
    LOG_DEBUG("SWCRenderer InitLine with vertices count : {}, indices count : {}, patch_id : {}",
              vertices.size(), indices.size(), patch_id);
    assert(patches.count(patch_id) == 0);
    assert(preserved_indices_per_patch && preserved_vertices_per_patch);
    auto& patch = patches[patch_id];
    auto& draw_patch = patch.draw_patch;
    auto& draw_tag = patch.draw_tag;
    draw_patch.model = model;
    draw_tag.model = model;
    for(auto& vert : vertices){
        patch.vertices_mp[vert] = patch.idx++;
    }
    int vsize = vertices.size(), isize = indices.size();
    draw_patch.loaded_vertices_count = vsize;
    draw_patch.loaded_indices_count = isize;
    draw_tag.loaded_vertices_count = vsize;
    // note vertices.size() == 0
    int vcnt = std::max(1, (vsize + preserved_vertices_per_patch - 1) / preserved_vertices_per_patch);
    int icnt = std::max(1, (isize + preserved_indices_per_patch - 1) / preserved_indices_per_patch);
    vsize = vcnt * preserved_vertices_per_patch;
    isize = icnt * preserved_indices_per_patch;
    draw_patch.vao.initialize_handle();
    draw_patch.vbo.initialize_handle();
    draw_patch.vbo.reinitialize_buffer_data(nullptr, vsize, GL_DYNAMIC_DRAW);
    draw_patch.ebo.initialize_handle();
    draw_patch.ebo.reinitialize_buffer_data(nullptr, isize, GL_DYNAMIC_DRAW);
    draw_patch.vao.bind_index_buffer(draw_patch.ebo);
    draw_patch.vao.bind_vertex_buffer_to_attrib(attrib_var_t<Vertex>(0), draw_patch.vbo, 0);
    draw_patch.vao.enable_attrib(attrib_var_t<Vertex>(0));
    draw_tag.vao.initialize_handle();
    draw_tag.vbo.initialize_handle();
    draw_tag.vbo.reinitialize_buffer_data(nullptr, vsize, GL_DYNAMIC_DRAW);
    draw_tag.vao.bind_vertex_buffer_to_attrib(attrib_var_t<Vertex>(0), draw_tag.vbo, 0);
    draw_tag.vao.enable_attrib(attrib_var_t<Vertex>(0));
    if(vertices.size() > 0){
        draw_patch.vbo.set_buffer_data(vertices.data(), 0, vertices.size());
        draw_tag.vbo.set_buffer_data(vertices.data(), 0, vertices.size());
    }
    if(indices.size() > 0)
        draw_patch.ebo.set_buffer_data(indices.data(), 0, indices.size());
}

void SWCRenderer::AddLine(const SWCRenderer::Vertex &vertex_a, const SWCRenderer::Vertex &vertex_b, size_t patch_id) {
    if(patches.count(patch_id) == 0){
        InitLine({}, {}, patch_id);
    }
    auto& patch = patches.at(patch_id);
    ExpandIfFull(patch.draw_patch);
    ExpandIfFull(patch.draw_tag);
    std::vector<Vertex> vtx;
    if(patch.vertices_mp.count(vertex_a) == 0){
        vtx.push_back(vertex_a);
        patch.vertices_mp[vertex_a] = patch.idx++;
    }
    if(patch.vertices_mp.count(vertex_b) == 0){
        vtx.push_back(vertex_b);
        patch.vertices_mp[vertex_b] = patch.idx++;
    }
    if(vtx.size() > 0){
        patch.draw_patch.vbo.set_buffer_data(vtx.data(), patch.draw_patch.loaded_vertices_count, vtx.size());
        patch.draw_tag.vbo.set_buffer_data(vtx.data(), patch.draw_tag.loaded_vertices_count, vtx.size());
    }
    patch.draw_patch.loaded_vertices_count += vtx.size();
    patch.draw_tag.loaded_vertices_count += vtx.size();

    uint32_t idxes[2] = {patch.vertices_mp.at(vertex_a), patch.vertices_mp.at(vertex_b)};
    patch.draw_patch.ebo.set_buffer_data(idxes, patch.draw_patch.loaded_indices_count, 2);
    patch.draw_patch.loaded_indices_count += 2;

}

void SWCRenderer::DeleteLine(size_t patch_id) {

}

void SWCRenderer::Draw(const mat4 &view, const mat4& proj, bool tag) {
    shader.bind();
    //todo: id mapping to color
    transform_params.model = mat4::identity();
    transform_params.view = view;
    transform_params.proj = proj;
    transform_params_buffer.set_buffer_data(&transform_params);
    transform_params_buffer.bind(0);
    for(auto& [id, patch] : patches){
        auto& draw_patch = patch.draw_patch;
        draw_patch.vao.bind();

        GL_EXPR(glDrawElements(GL_LINES, draw_patch.loaded_indices_count, GL_UNSIGNED_INT, nullptr));

        draw_patch.vao.unbind();
    }

    shader.unbind();

    if(!tag) return;

    pt_shader.bind();

    transform_params_buffer.bind(0);

    for(auto& [id, patch] : patches){
        auto& draw_tag = patch.draw_tag;
        draw_tag.vao.bind();

        GL_EXPR(glDrawArrays(GL_POINTS, 0, draw_tag.loaded_vertices_count));

        draw_tag.vao.unbind();
    }
    GL_EXPR(glDisable(GL_DEPTH_TEST));
    pt_shader.set_uniform_var("Sel", 1);
    if(sel_tags.loaded_vertices_count > 0){
        sel_tags.vao.bind();

        GL_EXPR(glDrawArrays(GL_POINTS, 0, sel_tags.loaded_vertices_count));

        sel_tags.vao.unbind();
    }
    pt_shader.set_uniform_var("Sel", 0);
    GL_EXPR(glEnable(GL_DEPTH_TEST));
    pt_shader.unbind();

}

void SWCRenderer::ExpandIfFull(DrawPatch& patch) {
    if(patch.loaded_vertices_count && patch.loaded_vertices_count % preserved_vertices_per_patch == 0){
        int new_vertices_count = patch.loaded_vertices_count + preserved_vertices_per_patch;
        vertex_buffer_t<Vertex> vbo;
        vbo.initialize_handle();
        vbo.reinitialize_buffer_data(nullptr, new_vertices_count, GL_DYNAMIC_DRAW);
        GL_EXPR(glCopyNamedBufferSubData(patch.vbo.handle(),
                                 vbo.handle(),
                                 0, 0,
                                 patch.loaded_vertices_count * sizeof(Vertex)));
        patch.vbo = std::move(vbo);
        patch.vao.bind_vertex_buffer_to_attrib(attrib_var_t<Vertex>(0), patch.vbo, 0);
        patch.vao.enable_attrib(attrib_var_t<Vertex>(0));
//        patch.loaded_vertices_count = new_vertices_count;
    }
    if(patch.loaded_indices_count && patch.loaded_indices_count % preserved_indices_per_patch == 0){
        int new_indices_count = patch.loaded_indices_count + preserved_indices_per_patch;
        index_buffer_t<uint32_t> ebo;
        ebo.initialize_handle();
        ebo.reinitialize_buffer_data(nullptr, new_indices_count, GL_DYNAMIC_DRAW);
        GL_EXPR(glCopyNamedBufferSubData(patch.ebo.handle(),
                                         ebo.handle(),
                                         0, 0,
                                         patch.loaded_indices_count * sizeof(uint32_t)));
        patch.ebo = std::move(ebo);
        patch.vao.bind_index_buffer(patch.ebo);
//        patch.loaded_indices_count = new_indices_count;
    }
}
void SWCRenderer::ExpandIfFull(DrawTag& tag){
    if(tag.loaded_vertices_count && tag.loaded_vertices_count % preserved_vertices_per_patch == 0){
        int new_vertices_count = tag.loaded_vertices_count + preserved_vertices_per_patch;
        vertex_buffer_t<Vertex> vbo;
        vbo.initialize_handle();
        vbo.reinitialize_buffer_data(nullptr, new_vertices_count, GL_DYNAMIC_DRAW);
        GL_EXPR(glCopyNamedBufferSubData(tag.vbo.handle(),
                                         vbo.handle(),
                                         0, 0,
                                         tag.loaded_vertices_count * sizeof(Vertex)));
        tag.vbo = std::move(vbo);
        tag.vao.bind_vertex_buffer_to_attrib(attrib_var_t<Vertex>(0), tag.vbo, 0);
        tag.vao.enable_attrib(attrib_var_t<Vertex>(0));
    }
}
void SWCRenderer::Set(SWCPointKey a, SWCPointKey b)
{
    pt_shader.bind();

    pt_shader.set_uniform_var("PickedID[0]", a);
    pt_shader.set_uniform_var("PickedID[1]", b);

    pt_shader.unbind();
}

