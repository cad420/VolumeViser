#version 460 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
in VS_OUT{
    vec3 color;
    int id;
}gs_in[];

uniform int PickedID[2];

layout(location = 0) out vec3 fColor;
layout(location = 1) flat out int fID;

void build_quad(vec4 position){
    fColor = gs_in[0].color;
    fID = gs_in[0].id;
    position = position / position.w;
    if(position.z < -1.f || position.z > 1.f) return;
    const float s = 0.012f ;
    gl_Position = vec4(position.xy, -1.0, 1.0) + vec4(-s, -s, 0.0, 0.0);
    EmitVertex();
    gl_Position = vec4(position.xy, -1.0, 1.0) + vec4(s, -s, 0.0, 0.0);
    EmitVertex();
    gl_Position = vec4(position.xy, -1.0, 1.0) + vec4(-s, s, 0.0, 0.0);
    EmitVertex();
    gl_Position = vec4(position.xy, -1.0, 1.0) + vec4(s, s, 0.0, 0.0);
    EmitVertex();
    EndPrimitive();
}

void main(){
    build_quad(gl_in[0].gl_Position);
}