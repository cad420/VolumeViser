#version 460 core
layout(location = 0) in vec3 iFragColor;
layout(location = 1) flat in int iFragID;
layout(location = 0) out vec4 oFragColor;
layout(location = 1) out int oFragID;
void main(){
    oFragColor = vec4(iFragColor, 1.0);
    oFragID = iFragID;
}