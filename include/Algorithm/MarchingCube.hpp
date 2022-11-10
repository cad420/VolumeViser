#pragma once

#include <Common/Common.hpp>
#include <Core/Renderer.hpp>

VISER_BEGIN



class MarchingCubeAlgoPrivate;

class MarchingCubeAlgo : public UnifiedRescBase{
public:
    struct MarchingCubeAlgoCreateInfo{

    };
    MarchingCubeAlgo(const MarchingCubeAlgoCreateInfo& info);

    ~MarchingCubeAlgo();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void BindVTexture(VTextureHandle handle, TextureUnit unit);

    void BindPTBuffer(PTBufferHandle handle);

    struct MarchingCubeAlgoParams{
        UInt3 shape;
        UInt3 origin;
        float isovalue;
    };
    void Run(const MarchingCubeAlgoParams& params);

private:
    std::unique_ptr<MarchingCubeAlgoPrivate> _;
};




VISER_END