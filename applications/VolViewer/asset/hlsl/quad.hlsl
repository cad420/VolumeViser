
struct VSOutput{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

VSOutput VSMain(uint vertexID : SV_VertexID){
    VSOutput output;
    output.texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(output.texCoord * float2(2, -2) + float2(-1, 1), 0.5, 1);
    return output;
}

Texture2D<float4> Color : register(t0);

SamplerState LinearSampler : register(s0);

float4 PSMain(VSOutput input) : SV_TARGET
{

    float4 color = Color.Sample(LinearSampler, input.texCoord);

    return color;
}