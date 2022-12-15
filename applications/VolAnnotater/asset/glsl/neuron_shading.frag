#version 460 core

layout(location = 0) in vec3 iFragPos;
layout(location = 1) in vec3 iFragNormal;

layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 1) uniform Params{
    vec3 LightDir;
    vec3 LightRadiance;
    vec3 Albedo;
    vec3 ViewPos;
};

void main(){
    vec3 view_dir = normalize((ViewPos - iFragPos));

    vec3 normal = normalize(iFragNormal);

    vec3 ambient = vec3(0.05);

    vec3 diffuse = max(0.0, dot(normal, view_dir)) * Albedo * LightRadiance;

    vec3 specular = 0.2f * pow(max(0.0, dot(normal, normalize(view_dir + LightDir))), 12.f) * Albedo * LightRadiance;

    vec3 color = ambient + diffuse + specular;

    color = pow(color, vec3(1.0 / 2.2));

    oFragColor = vec4(color, 1.0);

//    oFragColor = vec4(normalize(iFragNormal) * 0.5 + 0.5, 1.f);
}