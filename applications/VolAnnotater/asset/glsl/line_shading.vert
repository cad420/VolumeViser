#version 460 core
layout(location = 0) in vec4 iVertexR;

layout(std140, binding = 0) uniform Transform{
    mat4 Model;
    mat4 View;
    mat4 Proj;
};

void main(){
    vec4 view_pos = View * Model * vec4(iVertexR.xyz, 1.0);
//    view_pos /= view_pos.w;
//    view_pos.z = min(iVertexR.w + view_pos.z, 0.0);
    gl_Position = Proj * view_pos;
}