#include "Common.hpp"
#include "VolAnnotater.hpp"

//============================================================================================
void AppSettings::Initialize(const VolAnnotaterCreateInfo &info) {
    AppSettings::MaxHostMemBytes = info.max_host_mem_bytes;
    AppSettings::MaxRenderGPUMemBytes = info.max_render_gpu_mem_bytes;
    AppSettings::MaxComputeGPUMemBytes = info.max_compute_gpu_mem_bytes;
    AppSettings::RenderGPUIndex = info.render_gpu_index;
    AppSettings::ComputeGPUIndex = info.compute_gpu_index;
    AppSettings::MaxFixedHostMemBytes = info.max_fixed_host_mem_bytes;
    AppSettings::ThreadsGroupWorkerCount = info.threads_count;
    AppSettings::VTexCount = info.vtex_count;
    AppSettings::VTexShape = Int3(info.vtex_shape_x, info.vtex_shape_y, info.vtex_shape_z);
}

//============================================================================================

void ViserRescPack::Initialize() {

}

void ViserRescPack::LoadVolume(const std::string &filename) {

}

//============================================================================================

void VolRenderRescPack::Initialize() {

}

//============================================================================================

void SWCRescPack::Initialize() {

}

//============================================================================================

void SWC2MeshRescPack::Initialize() {

}

//============================================================================================
