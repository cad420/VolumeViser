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

    void AddNeuronMesh(const MeshData1& mesh, PatchID patch_id);

    void DeleteNeuronMesh(PatchID patch_id);

    void Draw(const mat4& view, const mat4& proj);

private:
    program_t shader;



};