#include <Algorithm/Voxelization.hpp>
#include "Common.hpp"
#include "VolAnnotater.hpp"
#include "SWCRenderer.hpp"
#include "NeuronRenderer.hpp"
#include <unordered_set>
#include <algorithm>

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

        vol_info.lod_vol_file_io[lod] = Handle<VolumeIOInterface>(ResourceType::Object, CreateVolumeFileByFileName(lod_path));
    }

    vol_priv_data.volume = NewHandle<GridVolume>(ResourceType::Object, vol_info);

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

    auto host_pool_uid = host_mem_mgr_ref.Invoke(&HostMemMgr::RegisterFixedHostMemMgr, host_pool_info);
    host_block_pool_ref = host_mem_mgr_ref.Invoke(&HostMemMgr::GetFixedHostMemMgrRef, host_pool_uid).LockRef();
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

    auto vtex_uid = render_gpu_mem_mgr_ref.Invoke(&GPUMemMgr::RegisterGPUVTexMgr, vtex_info);
    gpu_vtex_mgr_ref = render_gpu_mem_mgr_ref.Invoke(&GPUMemMgr::GetGPUVTexMgrRef, vtex_uid);
    gpu_pt_mgr_ref = gpu_vtex_mgr_ref.Invoke(&GPUVTexMgr::GetGPUPageTableMgrRef);
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
    crt_vol_renderer = NewHandle<CRTVolumeRenderer>(ResourceType::Object, renderer_info);

    framebuffer = NewHandle<FrameBuffer>(ResourceType::Buffer);


    vol_query_priv_data.query_info = NewHandle<CUDAHostBuffer>(ResourceType::Buffer,
                                                                    sizeof(float) * 8,
                                                                    cub::cu_memory_type::e_cu_host,
                                                                    _.render_gpu_mem_mgr_ref._get_ptr()->_get_cuda_context());
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
//    render_params.tf.tf_pts.pts[0.f] = Float4(0.f);
    render_params.tf.tf_pts.pts[0.25f] = Float4(0.f, 0.1f, 0.2f, 0.f);
    render_params.tf.tf_pts.pts[0.6f] = Float4(0.3f, 0.6f, 0.9f, 1.f);
    render_params.tf.tf_pts.pts[0.96f] = Float4(0.f, 0.5f, 1.f, 1.f);
//    render_params.tf.tf_pts.pts[1.f] = Float4(0.f);
    render_params.other.updated = true;
    render_params.other.ray_step = render_base_space * 0.5;
    render_params.other.max_ray_dist = 6.f;
    render_params.other.inv_tex_shape = Float3(1.f / AppSettings::VTexShape.x,
                                               1.f / AppSettings::VTexShape.y,
                                               1.f / AppSettings::VTexShape.z);
    crt_vol_renderer->SetRenderParams(render_params);

    auto vtexs = _.gpu_vtex_mgr_ref.Invoke(&GPUVTexMgr::GetAllTextures);
    for(auto& [unit, handle] : vtexs){
        crt_vol_renderer->BindVTexture(handle, unit);
    }

}

void VolRenderRescPack::UpdateTransferFunc(const std::vector<std::pair<float, Float4>>& pts){
    RenderParams render_params;
    render_params.tf.updated = true;
    for (auto &pt : pts)
    {
        render_params.tf.tf_pts.pts[pt.first] = pt.second;
    }
    crt_vol_renderer->SetRenderParams(render_params);
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
    swc_file = NewHandle<SWCFile>(viser::ResourceType::Object);

    swc_renderer = std::make_unique<SWCRenderer>(SWCRenderer::SWCRendererCreateInfo{});


}

void SWCRescPack::LoadSWCFile(const std::string &filename, Float3 ratio) {
    if(filename.empty()) return;
    try {
        swc_file->Open(filename, SWCFile::Read);

        auto swc_pts = swc_file->GetAllPoints();

        swc_file->Close();

        CreateSWC(filename);

        float s = (std::min)({ratio.x, ratio.y, ratio.z});

        for(auto& pt : swc_pts){
            pt.x *= ratio.x;
            pt.y *= ratio.y;
            pt.z *= ratio.z;
            pt.radius *= s;
            InsertSWCPoint(pt);
        }
        double dist_sum = 0.f;
        auto& swc = GetSelected().swc;
        for(auto& pt : swc_pts){
            auto& pnode = swc->GetNode(pt.pid);
            Float3 ppos = Float3(pnode.x, pnode.y, pnode.z);
            Float3 pos = Float3(pt.x, pt.y, pt.z);
            dist_sum += (ppos - pos).length();
        }
        LOG_INFO("swc total dist sum : {}", dist_sum);
        swc_priv_data.swc_draw_tree.Build(GetSelected().swc);
    }
    catch (const ViserFileOpenError& err) {
        LOG_ERROR("LoadSWCFile error : {}", err.what());
    }
}

void SWCRescPack::CreateSWC(const std::string& filename) {
    static int swc_count = 0;
    auto swc = NewHandle<SWC>(viser::ResourceType::Object);
    auto swc_uid = swc->GetUID();
    assert(CheckUnifiedRescUID(swc_uid));
    auto& swc_info = loaded_swc[swc_uid];
    swc_info.swc = std::move(swc);
    swc_info.name = "SWC_" + std::to_string(++swc_count);
    swc_info.filename = filename;

    SelectSWC(swc_uid);

    swc_priv_data.swc_draw_tree.swc = GetSelected().swc;
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

    ResetSWCRenderer();
//    swc_renderer->Reset();
//    auto lines = swc->PackLines();
//    for(auto& line : lines){
//        int n = line.size();
//        int root = swc->GetNodeRoot(line[0].id);
//        auto neuron_id = swc_priv_data.pt_to_neuron_mp[root];
//        for(int i = 1; i < n; i++){
//            vec4f prev_vert = vec4f(line[i - 1].x, line[i - 1].y, line[i - 1].z, vutil::intBitsToFloat(line[i - 1].id));
//            vec4f cur_vert = vec4f(line[i].x, line[i].y, line[i].z, vutil::intBitsToFloat(line[i].id));
//            swc_renderer->AddLine(prev_vert, cur_vert, neuron_id);
//        }
//    }

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
        if(pt.pid != 0){
//            swc_priv_data.last_picked_swc_pt_id = pt.pid;
            AddPickedSWCPoint(pt.pid);
        }

        pt.pid = swc_priv_data.last_picked_swc_pt_id;
    }
    if(swc_priv_data.available_swc_pt_ids.empty()){
        throw std::runtime_error("ERROR: Available SWC Point IDS Not Enough!!!");
    }

    SWCNeuronID id;
    if(pt.id == 0)
        id = *swc_priv_data.available_swc_pt_ids.begin();
    else
        id = pt.id;
    swc_priv_data.available_swc_pt_ids.erase(id);
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
        swc_renderer->InitLine({vec4f(pt.x, pt.y, pt.z, vutil::intBitsToFloat(pt.id))}, {}, neuron_id);
    }
    else{
        //与上一次选中的点形成一条线
        auto neuron_id = swc_priv_data.pt_to_neuron_mp.at(swc->GetNodeRoot(pt.pid));
        vec4f cur_vert = vec4f(pt.x, pt.y, pt.z, vutil::intBitsToFloat(pt.id));
        auto prev_node = swc->GetNode(pt.pid);
        auto prev_vert = vec4f(prev_node.x, prev_node.y, prev_node.z, vutil::intBitsToFloat(prev_node.id));
        swc_renderer->AddLine(prev_vert, cur_vert, neuron_id);
    }
    //更新最新一次选中的点
    //默认当成功插入一个点后 该点成为最新一次选中的点
//    swc_priv_data.last_picked_swc_pt_id = pt.id;
    AddPickedSWCPoint(pt.id);
}

void SWCRescPack::InsertInternalSWCPoint(SWC::SWCPoint pt)
{
    auto id1 = swc_priv_data.picked_swc_pt_q.back();
    auto id2 = swc_priv_data.picked_swc_pt_q.front();
    auto swc = GetSelected().swc;
    if(id1 <= 0 || id2 <= 0 || swc->GetNode(id1).pid != id2 && swc->GetNode(id2).pid != id1){
        LOG_DEBUG("Insert internal swc point should select two neighborhood points");
        return;
    }
    if(swc->GetNode(id1).pid == id2) std::swap(id1, id2);
    pt.pid = id1;

    SWCNeuronID id;
    if(pt.id == 0)
        id = *swc_priv_data.available_swc_pt_ids.begin();
    else
        id = pt.id;
    swc_priv_data.available_swc_pt_ids.erase(id);
    pt.id = id;

    swc->InsertNodeInternal(pt, id2);

    ResetSWCRenderer();
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
void SWCRescPack::AddPickedSWCPoint(SWCPointKey id){
    assert(id > 0);
    swc_priv_data.picked_swc_pt_q.push(id);
    if(swc_priv_data.picked_swc_pt_q.size() > swc_priv_data.picked_count){
        swc_priv_data.picked_swc_pt_q.pop();
    }
    assert(swc_priv_data.picked_swc_pt_q.size() == swc_priv_data.picked_count);
    swc_renderer->Set(swc_priv_data.picked_swc_pt_q.front(), swc_priv_data.picked_swc_pt_q.back());

    swc_priv_data.last_picked_swc_pt_id = id;

}
void SWCRescPack::SetSWCPointPickSize(int s)
{
    swc_priv_data.picked_count = s;
    while(swc_priv_data.picked_swc_pt_q.size() > s) swc_priv_data.picked_swc_pt_q.pop();
    if(!swc_priv_data.picked_swc_pt_q.empty() && swc_priv_data.picked_swc_pt_q.size() < s){
        auto t = swc_priv_data.picked_swc_pt_q.front();
        swc_priv_data.picked_swc_pt_q.pop();
        swc_priv_data.picked_swc_pt_q.push(0);
        swc_priv_data.picked_swc_pt_q.push(t);
    }
    if(!swc_priv_data.picked_swc_pt_q.empty())
        swc_renderer->Set(swc_priv_data.picked_swc_pt_q.front(), swc_priv_data.picked_swc_pt_q.back());
}
void SWCRescPack::UpdatePickedSWCSegmentPoints()
{
    swc_priv_data.swc_draw_tree.SelSegPoints(swc_priv_data.picked_swc_pt_q.front(), swc_priv_data.picked_swc_pt_q.back());
}
void SWCRescPack::AddPickedSWCSegPtsToRenderer()
{
    auto id1 = swc_priv_data.picked_swc_pt_q.front();
    auto id2 = swc_priv_data.picked_swc_pt_q.back();
    auto swc = GetSelected().swc;
    std::vector<Float4> vertices;
    for(auto& id : swc_priv_data.swc_draw_tree.draw_segment_points){
        if(id == id1 || id == id2) continue;
        auto& pt = swc->GetNode(id);
        vertices.push_back(Float4(pt.x, pt.y, pt.z, vutil::intBitsToFloat(pt.id)));
    }
    swc_renderer->AddVertex(vertices);
}
void SWCRescPack::ResetSWCRenderer(bool init)
{
    swc_renderer->Reset();
    swc_renderer->ClearVertex();
    if(init){
        if(!Selected()) return;
        auto swc = GetSelected().swc;
        auto lines = swc->PackLines();
        for(auto& line : lines){
            int n = line.size();
            int root = swc->GetNodeRoot(line[0].id);
            auto neuron_id = swc_priv_data.pt_to_neuron_mp[root];
            for(int i = 1; i < n; i++){
                vec4f prev_vert = vec4f(line[i - 1].x, line[i - 1].y, line[i - 1].z, vutil::intBitsToFloat(line[i - 1].id));
                vec4f cur_vert = vec4f(line[i].x, line[i].y, line[i].z, vutil::intBitsToFloat(line[i].id));
                swc_renderer->AddLine(prev_vert, cur_vert, neuron_id);
            }
        }
    }
}

//============================================================================================

void SWC2MeshRescPack::Initialize(ViserRescPack& _) {
    mesh_file = NewHandle<MeshFile>(ResourceType::Object);

    s2m_priv_data.segment_buffer = _.host_mem_mgr_ref.Invoke(&HostMemMgr::AllocPinnedHostMem,
                                                             ResourceType::Buffer,
                                                             MaxSegmentCount * sizeof(SWCSegment),
                                                             false);

    SWCVoxelizer::VoxelizerCreateInfo v_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(_.render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(_.host_mem_mgr_ref._get_ptr(), false)
    };
    swc_voxelizer = NewHandle<SWCVoxelizer>(ResourceType::Object, v_info);

    MarchingCubeAlgo::MarchingCubeAlgoCreateInfo mc_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(_.render_gpu_mem_mgr_ref._get_ptr(), false),
            .host_mem_mgr = Ref<HostMemMgr>(_.host_mem_mgr_ref._get_ptr(), false)
    };
    mc_algo = NewHandle<MarchingCubeAlgo>(ResourceType::Object, mc_info);

    neuron_renderer = std::make_unique<NeuronRenderer>(NeuronRenderer::NeuronRendererCreateInfo{});
}

void SWC2MeshRescPack::CreateBlockMesh(const BlockUID &uid) {
    s2m_priv_data.patch_mesh_mp[uid].status = Empty;
    s2m_priv_data.patch_mesh_mp[uid].mesh = NewHandle<Mesh>(ResourceType::Object);
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

        auto mesh = NewHandle<Mesh>(ResourceType::Object);

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
    auto mesh = NewHandle<Mesh>(ResourceType::Object);
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

    auto vtexs = _.gpu_vtex_mgr_ref.Invoke(&GPUVTexMgr::GetAllTextures);
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


using TreeNodeEleT = SWC::SWCPoint;
struct TreeNode{
    TreeNodeEleT data;
    int longest_depth = 0;
    float longest_path_length = 0.f;
    std::vector<TreeNode*> kids;
};
struct DrawTreeNode{
    Float2 pos;
    TreeNode* swc_node;
    DrawTreeNode* next = nullptr;//水平下一个子节点
    std::vector<DrawTreeNode*> up;//往上的子节点
    std::vector<DrawTreeNode*> down;//往下的子节点
};



void SWCDrawTree::Build(Handle<SWC> swc) {

    //传入的只能有一个根节点
    std::unordered_map<SWC::SWCPointKey, TreeNode*> mp;
    TreeNode* root = nullptr;
    int swc_pt_count = 0;
    for(auto it = swc->begin(); it != swc->end(); ++it){
        auto key = it->first;
        auto& pt = it->second;
        auto node = new TreeNode();
        node->data = pt;
        assert(mp.count(key) == 0);
        mp[key] = node;
        if(pt.pid == -1)
            root = node;
        swc_pt_count += 1;
    }
    for(auto& [key, node] : mp){
        auto pid = node->data.pid;
        if(pid != -1){
            mp.at(pid)->kids.push_back(node);
        }
    }
    VISER_WHEN_DEBUG(
    std::function<int(TreeNode*)> dfs_count = [&](TreeNode* node){
        if(!node) return 0;
        int cnt = 1;
        for(auto kid : node->kids){
            cnt += dfs_count(kid);
        }
        return cnt;
    };
    auto swc_pt_count_check = dfs_count(root);
    LOG_DEBUG("swc pt count {}, check {}", swc_pt_count, swc_pt_count_check);
    assert(swc_pt_count_check == swc_pt_count);
    )

    LOG_DEBUG("Build SWC Tree ok...");

    auto calc_dist = [](TreeNode* a, TreeNode* b){
        return Float3(a->data.x - b->data.x, a->data.y - b->data.y, a->data.z - b->data.z).length();
    };

    std::function<int(TreeNode*)> calc_longest_path_length = [&](TreeNode* node){
        if(!node) return 0;
        int len = 0;
        float dist = 0.f;
        for(auto kid : node->kids){
            auto l = calc_longest_path_length(kid);
            if(len < l){
                len = l;
                dist = kid->longest_path_length + calc_dist(node, kid);
            }
        }
        node->longest_depth = len + 1;
        node->longest_path_length = dist;
        return len + 1;
    };

    int longest_depth = calc_longest_path_length(root);
    LOG_DEBUG("SWC Tree Longest Path Length is {}", longest_depth);

    DrawTreeNode* draw_root = nullptr;

    const int MaxLevel = 16;
    std::map<int, float> levels;
    for(int level = 0; level <= MaxLevel; level++)
        levels[level] = levels[-level] = std::numeric_limits<float>::max();


    std::function<DrawTreeNode*(TreeNode*, float, float, int)> calc_2d = [&](TreeNode* node, float x, float y, int base)->DrawTreeNode*{
        if(!node) return nullptr;

        auto draw_node = new DrawTreeNode();
        draw_node->swc_node = node;
        draw_node->pos = Float2(x, y);
        //降序排序 按最长路径的节点个数 路径的长度可能不是最长的 没关系
        std::sort(node->kids.begin(), node->kids.end(), [](TreeNode* a, TreeNode* b){
            return a->longest_depth > b->longest_depth;
        });
        int n = node->kids.size();
        if(n == 0){
            //没有子节点

        }
        else{
            //只有一个水平的子节点
            auto& swc_pt = node->data;
            auto& swc_pt1 = node->kids.front()->data;
            auto dist = Float3(swc_pt.x - swc_pt1.x, swc_pt.y - swc_pt1.y, swc_pt.z - swc_pt1.z).length();
            draw_node->next = calc_2d(node->kids.front(), x + dist, y, base);
            if(n > 1){

                auto search_level = [&](int level, float tx, float x)->bool{
                    if(tx < levels[level] - 0.001f){
                        levels[level] = x;
                        for(int i = 1; i < level; i ++)
                            levels[i] = std::min(x, levels[i]);
                        for(int i = -1; i > level; i--)
                            levels[i] = std::min(x, levels[i]);
                        return true;
                    }
                    else return false;
                };

                for(int i = 1; i < n; i++){
                    for(int level = std::abs(base) + 1; level <= MaxLevel; level++){
                        float nx = x + calc_dist(node, node->kids[i]);
                        float tx = nx + node->kids[i]->longest_path_length;
                        if(base >= 0 && search_level(level, tx, x)){
                            auto& t = draw_node->up.emplace_back();
                            t = calc_2d(node->kids[i], nx, level, level);
                            break;
                        }
                        if(base <= 0 && search_level(-level, tx, x)){
                            auto& t = draw_node->down.emplace_back();
                            t = calc_2d(node->kids[i], nx, -level, -level);
                            break;
                        }
                    }
                }

            }
        }

        return draw_node;
    };
    draw_root = calc_2d(root, 0.f, 0.f, 0);

    this->draw_tree_root = draw_root;

    this->draw_lines = CalcDrawLines();

    this->draw_points = CalcDrawSWCPoints();

    this->swc = std::move(swc);
}

std::vector<std::pair<Float2, Float2>> SWCDrawTree::CalcDrawLines() {
    std::vector<std::pair<Float2, Float2>> ret;

    std::function<void(DrawTreeNode*)> dfs = [&](DrawTreeNode* node){
        if(!node) return;
        if(node->next){
            ret.emplace_back(node->pos, node->next->pos);
            dfs(node->next);
        }
        if(!node->up.empty()){
            auto h = node->up.back()->pos.y - node->pos.y;
            auto x = node->pos.x;
            ret.emplace_back(node->pos, node->pos + Float2(0, h));
            for(auto kid : node->up){
                ret.emplace_back(Float2(x, kid->pos.y), kid->pos);
                dfs(kid);
            }
        }
        if(!node->down.empty()){
            auto h = node->down.back()->pos.y - node->pos.y;
            auto x = node->pos.x;
            ret.emplace_back(node->pos, node->pos + Float2(0, h));
            for(auto kid : node->down){
                ret.emplace_back(Float2(x, kid->pos.y), kid->pos);
                dfs(kid);
            }
        }
    };
    dfs(draw_tree_root);

    return ret;
}

std::vector<DrawPoint> SWCDrawTree::CalcDrawSWCPoints() {
    std::vector<DrawPoint> ret;
    int id = 0;

    std::function<void(DrawTreeNode*)> dfs = [&](DrawTreeNode* node){
        if(!node) return;
        auto& p = ret.emplace_back();
        p.pos = node->pos;
        p.draw_id = id++;
        p.pt_id = node->swc_node->data.id;
        dfs(node->next);
        for(auto kid : node->up) dfs(kid);
        for(auto kid : node->down) dfs(kid);
    };

    dfs(draw_tree_root);

    return ret;
}
void SWCDrawTree::SelSegPoints(SWCPointKey a, SWCPointKey b){
    assert(swc.IsValid());

    if(!swc->QueryNode(a) || !swc->QueryNode(b)){
        LOG_DEBUG("SelSegPoints with invalid a or b");
        return;
    }

    if(!swc->CheckConnection(a, b)){
        LOG_ERROR("SelSegPoints: a and b are not connected");
    }
    draw_segment_points.clear();

    int p = swc->GetFirstCommonRoot(a, b);
    if(p == b){
        std::swap(a, b);
        p = a;
    }
    if(p == a){
        for(auto it = swc->begin(); it != swc->end(); ++it){
            bool ok1 = swc->IsRoot(a, it->first);
            bool ok2 = swc->IsRoot(it->first, b);
            if(ok1 && ok2){
                draw_segment_points.insert(it->first);
            }
        }
    }
    else{
        for(auto it = swc->begin(); it != swc->end(); ++it){
            bool ok1 = swc->IsRoot(p, it->first);
            bool ok2 = swc->IsRoot(it->first, a);
            ok2 = ok2 || swc->IsRoot(it->first, b);
            if(ok1 && ok2){
                draw_segment_points.insert(it->first);
            }
        }
    }

}
