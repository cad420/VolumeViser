#include <Algorithm/Voxelization.hpp>
#include "Common.hpp"
#include "VolAnnotater.hpp"
#include "SWCRenderer.hpp"
#include "NeuronRenderer.hpp"
#include <unordered_set>

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

    vol_priv_data.block_length = volume_desc.block_length;

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

//    ComputeUpBoundLOD(lod, render_base_space, 960, 540, vutil::deg2rad(40.f));
//    ComputeDefaultLOD(lod, (float)volume_desc.block_length * _.vol_priv_data.volume_space_ratio * render_base_space);
//    lod.LOD[_.vol_priv_data.max_lod] = std::numeric_limits<float>::max();
    UpdateDefaultLOD(_, lod_ratio);

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

void VolRenderRescPack::UpdateUpBoundLOD(ViserRescPack& _, float fov_rad, float ratio) {
    if(ratio < 1.f){
        LOG_WARN("UpBoundLOD ratio should greater/equal than 1.f");
    }
    auto volume_desc = _.vol_priv_data.volume->GetDesc();
    ComputeUpBoundLOD(lod, render_base_space, framebuffer->frame_width, framebuffer->frame_height, fov_rad);
    lod.LOD[_.vol_priv_data.max_lod] = std::numeric_limits<float>::max();
    for(int i = 0; i < _.vol_priv_data.max_lod; i++){
        lod.LOD[i] *= ratio;
    }
}

void VolRenderRescPack::UpdateDefaultLOD(ViserRescPack& _, float ratio) {
    auto volume_desc = _.vol_priv_data.volume->GetDesc();
    ComputeDefaultLOD(lod, (float)volume_desc.block_length * _.vol_priv_data.volume_space_ratio * render_base_space);
    lod.LOD[_.vol_priv_data.max_lod] = std::numeric_limits<float>::max();
    for(int i = 0; i < _.vol_priv_data.max_lod; i++){
        lod.LOD[i] *= ratio;
    }
}

std::vector<BlockUID> VolRenderRescPack::ComputeIntersectBlocks(const std::vector<SWC::SWCPoint> &pts) {
    std::vector<BlockUID> ret;
#define MINIMUM_INTERSECT_BLOCKS
#ifndef MINIMUM_INTERSECT_BLOCKS
    BoundingBox3D box;
    for(auto& pt : pts){
        box |= vec3f(pt.x - pt.radius, pt.y - pt.radius, pt.z - pt.radius);
        box |= vec3f(pt.x + pt.radius, pt.y + pt.radius, pt.z + pt.radius);
    }

    ComputeIntersectedBlocksWithBoundingBox(ret,
                                            render_vol.lod0_block_length_space,
                                            render_vol.lod0_block_dim,
                                            render_vol.volume_bound,
                                            box);
#else
    std::unordered_set<BlockUID> st;
    std::unordered_map<SWC::SWCPointKey, SWC::SWCPoint> mp;
    std::vector<BlockUID> tmp;
    for(auto& pt : pts) mp[pt.id] = pt;
    for(auto& pt : pts){
        BoundingBox3D box;
        if(mp.count(pt.pid)){
            auto& pt_a = mp.at(pt.pid);
            box |= vec3f(pt_a.x - pt_a.radius, pt_a.y - pt_a.radius, pt_a.z - pt_a.radius);
            box |= vec3f(pt_a.x + pt_a.radius, pt_a.y + pt_a.radius, pt_a.z + pt_a.radius);
        }
        box |= vec3f(pt.x - pt.radius, pt.y - pt.radius, pt.z - pt.radius);
        box |= vec3f(pt.x + pt.radius, pt.y + pt.radius, pt.z + pt.radius);
        ComputeIntersectedBlocksWithBoundingBox(tmp,
                                                render_vol.lod0_block_length_space,
                                                render_vol.lod0_block_dim,
                                                render_vol.volume_bound,
                                                box);
        for(auto& b : tmp)
            st.insert(b);
        tmp.clear();
    }
    for(auto& b : st)
        ret.push_back(b);
#endif
    return ret;
}

//============================================================================================

void SWCRescPack::Initialize() {
    swc_file = NewHandle<SWCFile>(viser::RescAccess::Unique);

    swc_renderer = std::make_unique<SWCRenderer>(SWCRenderer::SWCRendererCreateInfo{});
}

void SWCRescPack::LoadSWCFile(const std::string &filename) {
    if(filename.empty()) return;
    try {
        swc_file->Open(filename, SWCFile::Read);

        auto swc_pts = swc_file->GetAllPoints();

        swc_file->Close();

        CreateSWC(filename);

        for(auto& pt : swc_pts){
            pt.x *= 0.00032f;
            pt.y *= 0.00032f;
            pt.z *= 0.00032f;
            pt.radius *= 0.001f;
            InsertSWCPoint(pt);
        }
    }
    catch (const ViserFileOpenError& err) {
        LOG_ERROR("LoadSWCFile error : {}", err.what());
    }
}

void SWCRescPack::CreateSWC(const std::string& filename) {
    static int swc_count = 0;
    auto swc = NewHandle<SWC>(viser::RescAccess::Shared);
    auto swc_uid = swc->GetUID();
    assert(CheckUnifiedRescUID(swc_uid));
    auto& swc_info = loaded_swc[swc_uid];
    swc_info.swc = std::move(swc);
    swc_info.name = "SWC_" + std::to_string(++swc_count);
    swc_info.filename = filename;

    SelectSWC(swc_uid);
}

void SWCRescPack::DeleteSelSWC() {
    if(!Selected()) return;
    //删除绑定的mesh uid
    DeleteSWCMesh();

    loaded_swc.erase(selected_swc_uid);

    selected_swc_uid = INVALID_RESC_ID;
}

void SWCRescPack::SelectSWC(SWCUID swc_id) {
    if(!CheckUnifiedRescUID(swc_id)){
        LOG_ERROR("Select SWC with invalid swc id");
        return;
    }

    selected_swc_uid = swc_id;

    swc_priv_data.Reset();

    swc_renderer->Reset();

    auto& swc = loaded_swc.at(swc_id).swc;

    auto roots = swc->GetAllRootIDs();
    for(auto root : roots){
        if(swc_priv_data.available_neuron_ids.empty()){
            throw std::runtime_error("ERROR: Available Neuron IDS Not Enough!!!");
        }
        swc_priv_data.pt_to_neuron_mp[root] = *swc_priv_data.available_neuron_ids.begin();
        swc_priv_data.available_neuron_ids.erase(swc_priv_data.available_neuron_ids.begin());
    }

    auto pts = swc->PackAll();
    for(auto& pt : pts){
        if(swc_priv_data.available_swc_pt_ids.empty()){
            throw std::runtime_error("ERROR: Available SWC Point IDS Not Enough!!!");
        }
        swc_priv_data.available_swc_pt_ids.erase(pt.id);
    }

    auto lines = swc->PackLines();
    for(auto& line : lines){
        int n = line.size();
        int root = swc->GetNodeRoot(line[0].id);
        auto neuron_id = swc_priv_data.pt_to_neuron_mp[root];
        for(int i = 1; i < n; i++){
            vec4f prev_vert = vec4f(line[i - 1].x, line[i - 1].y, line[i - 1].z, line[i - 1].radius);
            vec4f cur_vert = vec4f(line[i].x, line[i].y, line[i].z, line[i].radius);
            swc_renderer->AddLine(prev_vert, cur_vert, neuron_id);
        }
    }

    //如果有绑定mesh uid 那么调用切换函数
    if(swc_mesh_mp.count(selected_swc_uid))
        on_swc_selected(swc_mesh_mp.at(selected_swc_uid));
}

void SWCRescPack::InsertSWCPoint(SWC::SWCPoint pt) {
    assert(CheckUnifiedRescUID(selected_swc_uid));

    auto& swc = loaded_swc.at(selected_swc_uid).swc;

    if(swc_priv_data.last_picked_swc_pt_id == SWC::INVALID_SWC_KEY){
        pt.pid = -1;
    }
    else{
        pt.pid = swc_priv_data.last_picked_swc_pt_id;
    }
    if(swc_priv_data.available_swc_pt_ids.empty()){
        throw std::runtime_error("ERROR: Available SWC Point IDS Not Enough!!!");
    }
    auto id = *swc_priv_data.available_swc_pt_ids.begin();
    swc_priv_data.available_swc_pt_ids.erase(swc_priv_data.available_swc_pt_ids.begin());
    pt.id = id;

    swc->InsertNodeLeaf(pt);

    if(pt.pid == -1){
        //作为一条新的神经元的根节点
        if(swc_priv_data.available_neuron_ids.empty()){
            throw std::runtime_error("ERROR: Available Neuron IDS Not Enough!!!");
        }
        auto neuron_id = *swc_priv_data.available_neuron_ids.begin();
        swc_priv_data.available_neuron_ids.erase(swc_priv_data.available_neuron_ids.begin());
        swc_priv_data.pt_to_neuron_mp[pt.id] = neuron_id;
        swc_renderer->InitLine({vec4f(pt.x, pt.y, pt.z, pt.radius)}, {}, neuron_id);
    }
    else{
        //与上一次选中的点形成一条线
        auto neuron_id = swc_priv_data.pt_to_neuron_mp.at(swc->GetNodeRoot(pt.pid));
        vec4f cur_vert = vec4f(pt.x, pt.y, pt.z, pt.radius);
        auto prev_node = swc->GetNode(pt.pid);
        auto prev_vert = vec4f(prev_node.x, prev_node.y, prev_node.z, prev_node.radius);
        swc_renderer->AddLine(prev_vert, cur_vert, neuron_id);
    }
    //更新最新一次选中的点
    //默认当成功插入一个点后 该点成为最新一次选中的点
    swc_priv_data.last_picked_swc_pt_id = pt.id;

}

void SWCRescPack::SaveSWCToFile(){
    try{
        if(!CheckUnifiedRescUID(selected_swc_uid)){
            LOG_ERROR("Current Selected SWC Invalid");
            return;
        }

        auto& swc_info = loaded_swc.at(selected_swc_uid);

        if(swc_info.filename.empty()){
            LOG_ERROR("Save SWC File with empty filename");
            return;
        }

        ExportSWCToFile(swc_info.filename);
    }
    catch (const ViserFileOpenError& err) {
        LOG_ERROR("SaveSWCToFile error : {}", err.what());
    }
}

void SWCRescPack::ExportSWCToFile(const std::string &filename) {
    try{
        if(!CheckUnifiedRescUID(selected_swc_uid)){
            LOG_ERROR("Current Selected SWC Invalid");
            return;
        }

        auto& swc_info = loaded_swc.at(selected_swc_uid);

        swc_file->Open(filename, SWCFile::Write);

        auto pts = swc_info.swc->PackAll();
        sort(pts.begin(), pts.end(), [](const auto& a ,const auto& b){
            return a.id < b.id;
        });

        swc_file->WritePoints(pts);

        swc_file->Close();

        swc_info.filename = filename;

        LOG_INFO("Successfully save swc file : {}", swc_info.filename);

    }
    catch (const ViserFileOpenError& err) {
        LOG_ERROR("ExportSWCToFile error : {}", err.what());
    }
}

void SWCRescPack::BindMeshToSWC(MeshUID mesh_id) {
    assert(Selected());
    swc_mesh_mp[selected_swc_uid] = mesh_id;

}
void SWCRescPack::DeleteSWCMesh() {
    assert(Selected());
    swc_mesh_mp.erase(selected_swc_uid);

}

void SWCRescPack::Commit() {
    assert(Selected());
    loaded_swc.at(selected_swc_uid).swc->Commit();
}

//============================================================================================

void SWC2MeshRescPack::Initialize(ViserRescPack& _) {
    mesh_file = NewHandle<MeshFile>(RescAccess::Unique);

    s2m_priv_data.segment_buffer = _.host_mem_mgr_ref->AllocPinnedHostMem(RescAccess::Shared,
                                                                          MaxSegmentCount * sizeof(SWCSegment),
                                                                          false);

    SWCVoxelizer::VoxelizerCreateInfo v_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(_.render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(_.host_mem_mgr_ref._get_ptr(), false)
    };
    swc_voxelizer = NewHandle<SWCVoxelizer>(RescAccess::Unique, v_info);

    MarchingCubeAlgo::MarchingCubeAlgoCreateInfo mc_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(_.render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(_.host_mem_mgr_ref._get_ptr(), false)
    };
    mc_algo = NewHandle<MarchingCubeAlgo>(RescAccess::Unique, mc_info);

    neuron_renderer = std::make_unique<NeuronRenderer>(NeuronRenderer::NeuronRendererCreateInfo{});
}

void SWC2MeshRescPack::CreateBlockMesh(const BlockUID &uid) {
    s2m_priv_data.patch_mesh_mp[uid].status = Empty;
    s2m_priv_data.patch_mesh_mp[uid].mesh = NewHandle<Mesh>(RescAccess::Shared);
}

void SWC2MeshRescPack::UpdateBlockMesh(const BlockUID &uid, Handle<Mesh> mesh) {
    if(!QueryBlockMesh(uid)){
        CreateBlockMesh(uid);
    }
    s2m_priv_data.patch_mesh_mp.at(uid).mesh = std::move(mesh);
    s2m_priv_data.patch_mesh_mp.at(uid).status = Updated;
}

void SWC2MeshRescPack::SetBlockMeshStatus(const BlockUID &uid, SWC2MeshRescPack::BlockMeshStatus status) {
    if(!QueryBlockMesh(uid)){
        CreateBlockMesh(uid);
    }
    s2m_priv_data.patch_mesh_mp.at(uid).status = status;
}

void SWC2MeshRescPack::MergeAllBlockMesh() {
    if(!Selected()){
        LOG_ERROR("MergeAllBlockMesh but not select valid mesh");
        return;
    }
    auto& merged_mesh = loaded_mesh.at(selected_mesh_uid);
    std::vector<Handle<Mesh>> res;
    for(auto& [uid, block_mesh] : s2m_priv_data.patch_mesh_mp){
        if(block_mesh.status != Updated){
            LOG_ERROR("merge block mesh with status is not Updated but is : {}",
                      block_mesh.status == Empty ? "Empty" : "Modified");
        }
        res.push_back(block_mesh.mesh);
    }
    merged_mesh.mesh = Mesh::Merge(res);

    SetMeshStatus(Merged);
}

void SWC2MeshRescPack::SetMeshStatus(MeshStatus status) {
    mesh_status = status;
    MeshUpdated();
}

void SWC2MeshRescPack::MeshUpdated() {
    if(!Selected()) return;
    neuron_renderer->Reset();
    if(mesh_status == None) return;
    else if(mesh_status == Merged){
        auto& mesh_info = loaded_mesh.at(selected_mesh_uid);
        neuron_renderer->AddNeuronMesh(mesh_info.mesh->GetPackedMeshData(),
                                       mesh_info.mesh->GetUID());
    }
    else if(mesh_status == Blocked){
        for(auto& [uid, block_mesh] : s2m_priv_data.patch_mesh_mp){
            if(block_mesh.status == Empty){
                LOG_ERROR("Update render block mesh but status is not Updated but is {}",
                          block_mesh.status == Empty ? "Empty" : "Modified");
                continue;
            }
            neuron_renderer->AddNeuronMesh(block_mesh.mesh->GetPackedMeshData(), uid.ToUnifiedRescUID());
        }
    }
    else{
        assert(false);
    }
}

void SWC2MeshRescPack::LoadMeshFile(const std::string &filename) {
    if(filename.empty()) return;
    try{

        mesh_file->Open(filename, MeshFile::Read);

        auto mesh_data = mesh_file->GetMesh();

        mesh_file->Close();

        CreateMesh("", filename);

        auto mesh = NewHandle<Mesh>(RescAccess::Shared);

        int idx = 0;
        for(auto& shape : mesh_data){
            mesh->Insert(shape, idx++);
        }
        mesh->MergeShape();

        UpdateMesh(selected_mesh_uid, std::move(mesh));
    }
    catch (const ViserFileOpenError& err) {
        LOG_ERROR("LoadMeshFile error : {}", err.what());
    }
}

void SWC2MeshRescPack::CreateMesh(const std::string& name, const std::string &filename) {
    static int mesh_count = 0;
    mesh_count += 1;
    auto mesh = NewHandle<Mesh>(RescAccess::Shared);
    auto mesh_uid = mesh->GetUID();
    auto& mesh_info = loaded_mesh[mesh_uid];
    mesh_info.mesh = std::move(mesh);
    mesh_info.filename = filename;
    mesh_info.name = name.empty() ? "Neuron_Mesh_" + std::to_string(mesh_count) : name;

    Select(mesh_uid);

    //新建的mesh初始状态应该是Blocked
    SetMeshStatus(Blocked);
}

void SWC2MeshRescPack::Select(MeshUID mesh_id) {
    if(loaded_mesh.count(mesh_id) == 0){
        LOG_ERROR("Select mesh with invalid uid");
        return;
    }
    selected_mesh_uid = mesh_id;
    mesh_status = Merged;

    //清除block mesh
    ResetBlockedMesh();

    //重新上传渲染数据
    MeshUpdated();
}

void SWC2MeshRescPack::ResetBlockedMesh() {
    s2m_priv_data.patch_mesh_mp.clear();

}

void SWC2MeshRescPack::SaveMeshToFile() {
    if(!Selected()){
        LOG_ERROR("SaveMeshToFile with invalid selected mesh uid");
        return;
    }

    if(loaded_mesh.at(selected_mesh_uid).filename.empty()){
        LOG_ERROR("Save Mesh File with empty filename");
        return;
    }

    ExportMeshToFile(loaded_mesh.at(selected_mesh_uid).filename);
}

void SWC2MeshRescPack::ExportMeshToFile(const std::string &filename) {
    if(!Selected()){
        LOG_ERROR("ExportMeshToFile with invalid selected mesh uid");
        return;
    }
    try{
        auto& mesh_info = loaded_mesh.at(selected_mesh_uid);

        mesh_file->Open(filename, MeshFile::Write);

        mesh_file->WriteMeshData(mesh_info.mesh->GetPackedMeshData());

        mesh_file->Close();

        mesh_info.filename = filename;
    }
    catch (const ViserFileOpenError& err) {
        LOG_ERROR("ExportMeshFile error : {}", err.what());
    }
}

void SWC2MeshRescPack::UpdateMesh(MeshUID uid, Handle<Mesh> mesh) {
    if(loaded_mesh.count(uid) == 0){
        LOG_ERROR("UpdateMesh with invalid uid");
        return;
    }
    loaded_mesh.at(uid).mesh = std::move(mesh);

    if(uid == selected_mesh_uid)
        MeshUpdated();
}

void SWC2MeshRescPack::OnVolumeLoaded(ViserRescPack& _, VolRenderRescPack& __) {
    auto volume_desc = _.vol_priv_data.volume->GetDesc();
    VolumeParams vol_params;
    vol_params.block_length = volume_desc.block_length;
    vol_params.padding = volume_desc.padding;
    vol_params.voxel_dim = volume_desc.shape;
    vol_params.bound = {
            {0.f, 0.f, 0.f},
            Float3(vol_params.voxel_dim) * __.render_base_space * _.vol_priv_data.volume_space_ratio
    };
    vol_params.space = __.render_base_space * _.vol_priv_data.volume_space_ratio;

    swc_voxelizer->SetVolume(vol_params);

    mc_algo->SetVolume(vol_params);

    auto vtexs = _.gpu_vtex_mgr_ref->GetAllTextures();
    for(auto& [unit, handle] : vtexs){
        swc_voxelizer->BindVTexture(handle, unit);

        mc_algo->BindVTexture(handle, unit);
    }

}

void SWC2MeshRescPack::UpdateAllBlockMesh() {
    for(auto& [id, block_mesh] : s2m_priv_data.patch_mesh_mp){
        if(block_mesh.mesh->Empty())
            block_mesh.status = Empty;
        else
            block_mesh.status = Updated;
    }
}

void SWC2MeshRescPack::SmoothMesh(float lambda, float mu, int iterations) {
    if(!Selected() || mesh_status != Merged){
        LOG_ERROR("SmoothMesh must set selected and merged before");
        return;
    }
    GetSelected().mesh->Smooth(lambda, mu, iterations);

    MeshUpdated();
}

//============================================================================================