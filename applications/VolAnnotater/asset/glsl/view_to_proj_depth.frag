#version 460 core

layout(location = 0) in vec2 iUV;
layout(location = 1) in vec2 iScreenCoord;

layout(binding = 0) uniform sampler2D ViewDepth;

layout(binding = 0, std140) uniform Params{
    mat4 Proj;
    float Fov;
    float WOverH;
};
//layout(location = 0) out vec4 oFragColor;

void main(){
    vec2 iScreenCoord;
    float view_depth = texture(ViewDepth, iUV).r;
    float y_scale = view_depth * tan(Fov * 0.5);
    float x_scale = y_scale * WOverH;
    vec2 xy = vec2(x_scale, y_scale) * iScreenCoord;
    vec4 ndc = Proj * vec4(xy, view_depth, 1.0);
    gl_FragDepth = ndc.z / ndc.w * 0.5 + 0.5;
//    oFragColor = vec4(vec3(gl_FragDepth), 1.0);
}