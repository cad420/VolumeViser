#include <Core/GPUVTexMgr.hpp>
#include <Core/GPUPageTableMgr.hpp>

VISER_BEGIN

class GPUVTexMgrPrivate{

};

GPUVTexMgr::GPUVTexMgr(const GPUVTexMgrCreateInfo &info) {

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


VISER_END

