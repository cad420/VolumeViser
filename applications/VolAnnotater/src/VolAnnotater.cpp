#include "VolAnnotater.hpp"
#include "Common.hpp"

//标注系统的窗口绘制任务交给OpenGL，如果有多个显卡，其余的显卡可以用于网格重建任务
class VolAnnotaterApp : public gl_app_t{
public:
    using gl_app_t::gl_app_t;

    void initialize() override {
        auto& resc_ins = ResourceMgr::GetInstance();

        auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                              .MaxMemBytes = 32ull << 30,
                                                              .DeviceIndex = -1});
        host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

        auto render_gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                          .MaxMemBytes = 20ull << 30,
                                                          .DeviceIndex = 0});
        render_gpu_mem_mgr_ref = resc_ins.GetGPURef(render_gpu_resc_uid);

        std::unordered_map<uint32_t, std::string> lod_volume_files;
        GridVolume::GridVolumeCreateInfo vol_info{.min_lod = 1024, .max_lod = 0};
        vol_info.host_mem_mgr_uid = host_mem_mgr_uid;
        vol_info.gpu_mem_mgr_uid = render_gpu_resc_uid;
        for(auto& item : lod_volume_files){
            vol_info.min_lod = std::min(vol_info.min_lod, item.first);
            vol_info.max_lod = std::max(vol_info.max_lod, item.first);
            vol_info.lod_vol_file_io[item.first] = Handle<VolumeIOInterface>(RescAccess::Unique, std::make_shared<EBVolumeFile>(item.second));
        }
        volume = NewHandle<GridVolume>(RescAccess::Unique, vol_info);

        auto vol_desc = volume->GetDesc();

        size_t block_size = (vol_desc.block_length + vol_desc.padding * 2) * vol_desc.bits_per_sample
                * vol_desc.samples_per_voxel / 8;

        size_t block_num = (24ull << 30) / block_size;

        FixedHostMemMgr::FixedHostMemMgrCreateInfo host_block_pool_info{
            .host_mem_mgr = Ref<HostMemMgr>(host_mem_mgr_ref._get_ptr(), false),
            .fixed_block_size = block_size,
            .fixed_block_num = block_num
        };
        auto host_block_pool_uid = host_mem_mgr_ref->RegisterFixedHostMemMgr(host_block_pool_info);
        host_block_pool_ref = host_mem_mgr_ref->GetFixedHostMemMgrRef(host_block_pool_uid);

        GPUMemMgr::GPUVTexMgrCreateInfo tex_info{
            .gpu_mem_mgr = Ref<GPUMemMgr>(render_gpu_mem_mgr_ref._get_ptr(), false),
            .vtex_count = 16, .vtex_shape = {1024, 1024, 1024}, .bits_per_sample = vol_desc.bits_per_sample,
            .samples_per_channel = vol_desc.samples_per_voxel, .vtex_block_length = (int)(vol_desc.block_length + vol_desc.padding * 2),
            .is_float = vol_desc.is_float, .exclusive = true
        };
        auto vtex_uid = render_gpu_mem_mgr_ref->RegisterGPUVTexMgr(tex_info);
        gpu_vtex_mgr_ref = render_gpu_mem_mgr_ref->GetGPUVTexMgrRef(vtex_uid);
        gpu_pt_mgr_ref = gpu_vtex_mgr_ref->GetGPUPageTableMgrRef();




        thread_group.start(2);
    }

    void frame() override {
        handle_events();


    }

    void destroy() override{

    }
private:
    //标注系统不考虑并行，直接保存Ref就好
    Ref<HostMemMgr> host_mem_mgr_ref;
    Ref<GPUMemMgr> render_gpu_mem_mgr_ref;
//    Ref<GPUMemMgr> compute_gpu_mem_mgr_ref;

    //每次标注只针对一个体数据，可以运行过程中动态加载更换体数据对象
    Handle<GridVolume> volume;

    Ref<FixedHostMemMgr> host_block_pool_ref;

    Ref<GPUVTexMgr> gpu_vtex_mgr_ref;
    Ref<GPUPageTableMgr> gpu_pt_mgr_ref;

    vutil::thread_group_t thread_group;

    //由OpenGL资源注册得到的CUDA资源句柄
    FrameBuffer framebuffer;

    std::unique_ptr<CRTVolumeRenderer> crt_vol_renderer;

};

class VolAnnotaterPrivate{
public:


};

VolAnnotater::VolAnnotater(const VolAnnotaterCreateInfo &info) {

}

VolAnnotater::~VolAnnotater() {

}

void VolAnnotater::run() {
    auto app = std::make_unique<VolAnnotaterApp>(window_desc_t{
        .size = {1200, 900}, .title = "VolAnnotater"
    });

    app->run();

}
