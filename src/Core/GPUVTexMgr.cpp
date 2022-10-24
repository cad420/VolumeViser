#include <Core/GPUVTexMgr.hpp>
#include <Core/GPUPageTableMgr.hpp>

VISER_BEGIN

class GPUVTexMgrPrivate{
public:
    using GPUTexUnit = GPUVTexMgr::GPUTexUnit;
    std::unordered_map<GPUTexUnit, Handle<CUDATexture>> tex_mp;

    std::mutex g_mtx;

    UnifiedRescUID uid;
    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::GPUVTexMgr);
    }
};

GPUVTexMgr::GPUVTexMgr(const GPUVTexMgrCreateInfo &info) {
    _ = std::make_unique<GPUVTexMgrPrivate>();

    _->GenRescUID();
}

GPUVTexMgr::~GPUVTexMgr() {

}

void GPUVTexMgr::Lock() {

}

void GPUVTexMgr::UnLock() {

}

Ref<GPUPageTableMgr> GPUVTexMgr::GetGPUPageTableMgrRef() {
    return Ref<GPUPageTableMgr>();
}

void GPUVTexMgr::UploadBlockToGPUTex(Handle<CUDAHostBuffer> src, GPUVTexMgr::TexCoord dst) {

}

std::vector<GPUVTexMgr::GPUVTex> GPUVTexMgr::GetAllTextures() {
    return std::vector<GPUVTex>();
}


VISER_END

