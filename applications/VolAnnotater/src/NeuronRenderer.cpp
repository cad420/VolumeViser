#include "NeuronRenderer.hpp"

NeuronRenderer::NeuronRenderer(const NeuronRenderer::NeuronRendererCreateInfo &info) {
    shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("glsl/neuron_shading.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("glsl/neuron_shading.frag")
            );

    tf_params_buffer.initialize_handle();
    tf_params_buffer.reinitialize_buffer_data(nullptr, GL_DYNAMIC_DRAW);

    params_buffer.initialize_handle();
    params_buffer.reinitialize_buffer_data(&params, GL_DYNAMIC_DRAW);
}

NeuronRenderer::~NeuronRenderer() {

}

void NeuronRenderer::Reset() {
    patches.clear();
}

void NeuronRenderer::AddNeuronMesh(const MeshData0 &mesh, PatchID patch_id) {

    if(patches.count(patch_id)){
        DeleteNeuronMesh(patch_id);
    }
    assert(patches.count(patch_id) == 0);
    auto& patch = patches[patch_id];
    patch.vao.initialize_handle();
    patch.vbo.initialize_handle();
    patch.vbo.reinitialize_buffer_data(mesh.vertices.data(), mesh.vertices.size(), GL_STATIC_DRAW);
    patch.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0), patch.vbo, &Vertex::pos, 0);
    patch.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1), patch.vbo, &Vertex::normal, 1);
    patch.vao.enable_attrib(attrib_var_t<vec3f>(0));
    patch.vao.enable_attrib(attrib_var_t<vec3f>(1));
    patch.ebo.initialize_handle();
    patch.ebo.reinitialize_buffer_data(mesh.indices.data(), mesh.indices.size(), GL_STATIC_DRAW);
    patch.vao.bind_index_buffer(patch.ebo);

}

void NeuronRenderer::AddNeuronMesh(const MeshData1 &mesh, PatchID patch_id) {
    AddNeuronMesh(ConvertFrom(mesh), patch_id);
}

void NeuronRenderer::DeleteNeuronMesh(PatchID patch_id) {
    patches.erase(patch_id);
}

void NeuronRenderer::Begin(const mat4 &view, const mat4 &proj, const Float3& view_pos){
    shader.bind();

    tf_params.proj_view = proj * view;
    tf_params_buffer.set_buffer_data(&tf_params);
    tf_params_buffer.bind(0);

    params.view_pos = view_pos;
    params_buffer.set_buffer_data(&params);
    params_buffer.bind(1);
}

void NeuronRenderer::Draw(PatchID patch_id) {
    if(patches.count(patch_id) == 0) {
        LOG_ERROR("draw patch id is not existed: {}", patch_id);
        return;
    }

    auto& draw_patch = patches.at(patch_id);
    if(draw_patch.ebo.index_count() == 0) return;

    draw_patch.vao.bind();

    GL_EXPR(glDrawElements(GL_TRIANGLES, draw_patch.ebo.index_count(), GL_UNSIGNED_INT, nullptr));

    draw_patch.vao.unbind();
}


void NeuronRenderer::End(){
    shader.unbind();
}

void NeuronRenderer::Set(const Float3 &light_dir, const Float3 &light_radiance)
{
    params.light_dir = light_dir;
    params.light_radiance = light_radiance;
    params_buffer.set_buffer_data(&params);
}
