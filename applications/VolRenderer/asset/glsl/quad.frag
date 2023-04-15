#version 460 core
layout(binding = 0) uniform sampler2D Color;

layout(location = 0) in vec2 IUV;

layout(location = 0) out vec4 oFragColor;

void main() {
    oFragColor = texture(Color, IUV);
}
