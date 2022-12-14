#version 460 core
layout(location = 0) in vec4 iVertexR;

layout(std140, binding = 0) uniform Transform{
    mat4 Model;
    mat4 View;
    mat4 Proj;
};

uniform int PickedID[2];

out VS_OUT {
    vec3 color;
    int id;
} vs_out;

void main(){
    gl_Position = Proj * View * Model * vec4(iVertexR.xyz, 1.0);
    vs_out.id = floatBitsToInt(iVertexR.w);
    if(vs_out.id == PickedID[0] || vs_out.id == PickedID[1])
        vs_out.color = vec3(1.f, 1.f, 0.f);
    else
        vs_out.color = vec3(1.f, 0.f, 0.f);
}