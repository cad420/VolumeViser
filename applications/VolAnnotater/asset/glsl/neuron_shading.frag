#version 460 core

layout(location = 0) in vec3 iFragPos;
layout(location = 1) in vec3 iFragNormal;

layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 0) uniform Params{
    vec3 LightDir;
    vec3 LightRadiance;
    vec3 Albedo;
    vec3 ViewPos;
};

void main(){

    oFragColor = vec4(normalize(iFragNormal) * 0.5 + 0.5, 1.f);
}