#include "Common.hpp"
#include "VolAnnotater.hpp"

//============================================================================================
void AppSettings::Initialize(const VolAnnotaterCreateInfo &info) {
    LOG_TRACE("AppSettings Initialize...");
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
    LOG_TRACE("ViserRescPack Initialize start...");

    auto& resc_ins = ResourceMgr::GetInstance();
    auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                                 .MaxMemBytes = AppSettings::MaxHostMemBytes,
                                                                 .DeviceIndex = -1});

    host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

    auto render_gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                                    .MaxMemBytes = AppSettings::MaxRenderGPUMemBytes,
                                                                    .DeviceIndex = AppSettings::RenderGPUIndex});

    render_gpu_mem_mgr_ref = resc_ins.GetGPURef(render_gpu_resc_uid);

    if(AppSettings::ComputeGPUIndex != AppSettings::RenderGPUIndex){
        auto compute_gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                                  .MaxMemBytes = AppSettings::MaxComputeGPUMemBytes,
                                                                  .DeviceIndex = AppSettings::ComputeGPUIndex});

        compute_gpu_mem_mgr_ref = resc_ins.GetGPURef(compute_gpu_resc_uid);
    }
    else{
        compute_gpu_mem_mgr_ref = Ref<GPUMemMgr>(render_gpu_mem_mgr_ref._get_ptr(), false);
    }

    thread_group.start(AppSettings::ThreadsGroupWorkerCount);

    LOG_TRACE("ViserRescPack Initialize finish...");
}

void ViserRescPack::LoadVolume(const std::string &filename) {

    LOG_TRACE("Load Volume Lod File : {}", filename);

    std::ifstream in(filename);
    if(!in.is_open()){
        throw std::runtime_error("Open LOD Volume Data File failed : " + filename);
    }
    nlohmann::json j;
    in >> j;
    uint32_t levels = j.at("levels");
    if(levels > LevelOfDist::MaxLevelCount){
        LOG_WARN("Invalid levels for lod volume: {}", levels);
        levels = LevelOfDist::MaxLevelCount;
    }

    GridVolume::GridVolumeCreateInfo vol_info{
        .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false),
        .gpu_mem_mgr = Ref<GPUMemMgr>(render_gpu_mem_mgr_ref._get_ptr(), false),
        .levels = levels
    };
    for(uint32_t lod = 0; lod < levels; lod++){
        std::string lod_path = j.at("lod" + std::to_string(lod));
        LOG_TRACE("Load LOD({}) : {}", lod, lod_path);

        vol_info.lod_vol_file_io[lod] = Handle<VolumeIOInterface>(RescAccess::Shared, std::make_shared<EBVolumeFile>(lod_path));
    }

    vol_priv_data.volume = NewHandle<GridVolume>(RescAccess::Unique, vol_info);

    vol_priv_data.max_lod = levels - 1;

    auto volume_desc = vol_priv_data.volume->GetDesc();

    LOG_TRACE("Load LOD Volume({}) successfully", volume_desc.volume_name);
    VISER_WHEN_DEBUG(std::cout << volume_desc << std::endl)

    float volume_base_space = std::min({volume_desc.voxel_space.x,
                                        volume_desc.voxel_space.y,
                                        volume_desc.voxel_space.z});

    if(volume_base_space == 0){
        LOG_WARN("Volume base space is zero");
        vol_priv_data.volume_space_ratio = Float3(1.f);
    }
    else
        vol_priv_data.volume_space_ratio = Float3(volume_desc.voxel_space.x / volume_base_space,
                                    volume_desc.voxel_space.y / volume_base_space,
                                    volume_desc.voxel_space.z / volume_base_space);


    InitializeVolumeResc();
}

void ViserRescPack::InitializeVolumeResc() {
    auto volume_desc = vol_priv_data.volume->GetDesc();

    size_t block_size = (size_t)(volume_desc.block_length + volume_desc.padding * 2) * volume_desc.bits_per_sample
                        * volume_desc.samples_per_voxel / 8;
    block_size *= block_size * block_size;
    if(block_size == 0){
        throw std::runtime_error("Invalid block size equal to zero : " + std::to_string(block_size));
    }
    size_t block_num = AppSettings::MaxFixedHostMemBytes / block_size;
    if(block_num == 0){
        throw std::runtime_error("Invalid block size: " + std::to_string(block_size));
    }

    FixedHostMemMgr::FixedHostMemMgrCreateInfo host_pool_info{
            .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false),
            .fixed_block_size = block_size,
            .fixed_block_num = block_num
    };

    auto host_pool_uid = host_mem_mgr_ref->RegisterFixedHostMemMgr(host_pool_info);
    host_block_pool_ref = host_mem_mgr_ref->GetFixedHostMemMgrRef(host_pool_uid);
    LOG_TRACE("Successfully Create FixedHostMemMgr...");

    GPUMemMgr::GPUVTexMgrCreateInfo vtex_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false),
            .vtex_count = AppSettings::VTexCount,
            .vtex_shape = {AppSettings::VTexShape.x, AppSettings::VTexShape.y, AppSettings::VTexShape.z},
            .bits_per_sample = volume_desc.bits_per_sample,
            .samples_per_channel = volume_desc.samples_per_voxel,
            .vtex_block_length = (int)(volume_desc.block_length + volume_desc.padding * 2),
            .is_float = volume_desc.is_float, .exclusive = true
    };

    auto vtex_uid = render_gpu_mem_mgr_ref->RegisterGPUVTexMgr(vtex_info);
    gpu_vtex_mgr_ref = render_gpu_mem_mgr_ref->GetGPUVTexMgrRef(vtex_uid);
    gpu_pt_mgr_ref = gpu_vtex_mgr_ref->GetGPUPageTableMgrRef();
    LOG_TRACE("Successfully Create GPUVTexMgr...");

    LOG_TRACE("InitializeVolumeResc finish...");
}

//============================================================================================

void VolRenderRescPack::Initialize(ViserRescPack& _) {
    LOG_TRACE("VolRenderRescPack Initialize start...");

    CRTVolumeRenderer::CRTVolumeRendererCreateInfo renderer_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(_.render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(_.host_mem_mgr_ref._get_ptr(), false)
    };
    crt_vol_renderer = NewHandle<CRTVolumeRenderer>(RescAccess::Unique, renderer_info);

    framebuffer = NewGeneralHandle<FrameBuffer>(RescAccess::Unique);


    vol_query_priv_data.query_info = NewGeneralHandle<CUDAHostBuffer>(RescAccess::Unique,
                                                                    sizeof(float) * 8,
                                                                    cub::memory_type::e_cu_host,
                                                                    _.render_gpu_mem_mgr_ref->_get_cuda_context());
    vol_query_priv_data.query_info_view = vol_query_priv_data.query_info->view_1d<float>(sizeof(float)*8);


    LOG_TRACE("VolRenderRescPack Initialize finish...");
}

void VolRenderRescPack::OnVolumeLoaded(ViserRescPack& _) {
    auto volume_desc = _.vol_priv_data.volume->GetDesc();

    auto space = volume_desc.voxel_space;
    render_base_space = std::min({space.x, space.y, space.z});

    render_vol.lod0_block_dim = volume_desc.blocked_dim;
    render_vol.lod0_block_length_space = volume_desc.block_length * render_base_space * _.vol_priv_data.volume_space_ratio;
    render_vol.volume_bound = {
            Float3(0.f, 0.f, 0.f),
            Float3(volume_desc.shape.x * render_base_space * _.vol_priv_data.volume_space_ratio.x,
                   volume_desc.shape.y * render_base_space * _.vol_priv_data.volume_space_ratio.y,
                   volume_desc.shape.z * render_base_space * _.vol_priv_data.volume_space_ratio.z)
    };

    ComputeDefaultLOD(lod, (float)volume_desc.block_length * _.vol_priv_data.volume_space_ratio * render_base_space);
    lod.LOD[_.vol_priv_data.max_lod] = std::numeric_limits<float>::max();


    VolumeParams vol_params;
    vol_params.block_length = volume_desc.block_length;
    vol_params.padding = volume_desc.padding;
    vol_params.voxel_dim = volume_desc.shape;
    vol_params.bound = {
            {0.f, 0.f, 0.f},
            Float3(vol_params.voxel_dim) * render_base_space * _.vol_priv_data.volume_space_ratio
    };
    vol_params.space = render_base_space * _.vol_priv_data.volume_space_ratio;
    crt_vol_renderer->SetVolume(vol_params);

    RenderParams render_params;
    render_params.lod.updated = true;
    render_params.lod.leve_of_dist = lod;
    render_params.tf.updated = true;
    render_params.tf.tf_pts.pts[0.f] = Float4(0.f);
    render_params.tf.tf_pts.pts[0.25f] = Float4(0.f, 1.f, 0.5f, 0.f);
    render_params.tf.tf_pts.pts[0.6f] = Float4(1.f, 0.5f, 0.f, 1.f);
    render_params.tf.tf_pts.pts[0.96f] = Float4(1.f, 0.5f, 0.f, 1.f);
    render_params.tf.tf_pts.pts[1.f] = Float4(0.f);
    render_params.other.ray_step = render_base_space * 0.5;
    render_params.other.max_ray_dist = 6.f;
    render_params.other.inv_tex_shape = Float3(1.f / AppSettings::VTexShape.x,
                                               1.f / AppSettings::VTexShape.y,
                                               1.f / AppSettings::VTexShape.z);
    crt_vol_renderer->SetRenderParams(render_params);

    auto vtexs = _.gpu_vtex_mgr_ref->GetAllTextures();
    for(auto& [unit, handle] : vtexs){
        crt_vol_renderer->BindVTexture(handle, unit);
    }

}

//============================================================================================

void SWCRescPack::Initialize() {

}

//============================================================================================

void SWC2MeshRescPack::Initialize() {

}

//============================================================================================
