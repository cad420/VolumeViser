#version 460 core

layout(location = 0) in vec3 iVertexPos;
layout(location = 1) in vec3 iVertexNormal;

layout(location = 0) out vec3 oPos;
layout(location = 1) out vec3 oNormal;

layout(binding = 0, std140) uniform Transform{
    mat4 Model;
    mat4 ProjView;
};

void main(){
    vec4 pos = Model *vec4(iVertexPos, 1.f);
    gl_Position = ProjView * pos;
    oPos = vec3(pos.xyz / pos.w);
    oNormal = vec3(Model * vec4(iVertexNormal, 0.f));

}