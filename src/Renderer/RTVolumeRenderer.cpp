#undef UTIL_ENABLE_OPENGL

#include <Core/Renderer.hpp>
#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <unordered_set>
VISER_BEGIN

using BlockUID = GridVolume::BlockUID;

static Int3 DefaultVTexShape{1024, 1024, 1024};

namespace{

}

class RTVolumeRendererPrivate{
  public:
    struct{
        Ref<HostMemMgr> host_mem_mgr_ref;
        Ref<GPUMemMgr> gpu_mem_mgr_ref;
        Ref<GPUVTexMgr> gpu_vtex_mgr_ref;
        Ref<GPUPageTableMgr> gpu_pt_mgr_ref;
        Ref<FixedHostMemMgr> fixed_host_mem_mgr_ref;

        Handle<GridVolume> volume;
    };
    // data sts for loading blocks
    struct{
        bool async;
        std::vector<BlockUID> intersect_blocks;
        std::vector<GPUPageTableMgr::PageTableItem> block_infos;

    };

    struct{
        vutil::thread_group_t async_loading_queue;
        vutil::thread_group_t async_decoding_queue;
        vutil::thread_group_t async_transfer_queue;
//        std::vector<Handle<CUDAHostBuffer>> cached_host_blocks;
//        std::vector<Handle<CUDAHostBuffer>> missed_host_blocks;
//        std::unordered_set<BlockUID> cur_blocks_st;
        std::unordered_map<BlockUID, GPUPageTableMgr::PageTableItem> cur_block_infos_mp;
        std::map<int, std::vector<std::function<void()>>> tasks;
    }aq;

    void AQ_Init(){
        aq.async_loading_queue.start(6);
        aq.async_decoding_queue.start(3);
        aq.async_transfer_queue.start(2);
    }

    void AQ_Update(const std::vector<GPUPageTableMgr::PageTableItem>& cur_intersect_blocks){
        static std::vector<BlockUID> missed_blocks; missed_blocks.clear();
        aq.cur_block_infos_mp.clear();
        for(auto& [block_uid, tex_coord] : cur_intersect_blocks){
            if(!tex_coord.Missed()) continue;
            missed_blocks.emplace_back(block_uid);
            aq.cur_block_infos_mp[block_uid] = {block_uid, tex_coord};
        }

        auto task = aq.async_loading_queue.create_task([&]{
            std::vector<Handle<CUDAHostBuffer>> buffers;
            for(auto& block_uid : missed_blocks){
                buffers.emplace_back(fixed_host_mem_mgr_ref->GetBlock(block_uid.ToUnifiedRescUID()));
            }
            _AQ_AppendTask(buffers);
        });
        aq.async_loading_queue.submit(task);
    }


    void _AQ_AppendTask(std::vector<Handle<CUDAHostBuffer>> buffers){
        // if read lock, then transfer
        // if write lock, then loading and transfer
        for(auto& buffer : buffers){
            if(buffer.IsWriteLocked()){
                _AQ_AppendDecodingTask(buffer);
            }
            else if(buffer.IsReadLocked()){
                _AQ_AppendTransferTask(buffer);
            }
            else{
                assert(false);
            }
        }
    }

    void AQ_Commit(){
        // submit loading tasks
        static std::map<int, vutil::task_group_handle_t> task_groups; task_groups.clear();
        static std::vector<int> lods; lods.clear();
        for(auto& [lod, tasks] : aq.tasks){
              lods.emplace_back(lod);
              auto task_group = aq.async_decoding_queue.create_task();
              for(auto& task : tasks){
                  task_group->enqueue_task(std::move(task));
              }
              task_groups[lod] = std::move(task_group);
        }
        int lod_count = lods.size();
        for(int lod = 0; lod < lod_count - 1; ++lod){
            int first = lods[lod], second = lods[lod + 1];
            aq.async_decoding_queue.add_dependency(*task_groups[second], *task_groups[first]);
        }
        for(auto& [lod, task_group] : task_groups){
            aq.async_decoding_queue.submit(task_group);
        }

        // wait all transfer tasks finished
        aq.async_transfer_queue.wait_idle();
    }

    void _AQ_AppendDecodingTask(Handle<CUDAHostBuffer> buffer){
        auto block_uid = BlockUID(buffer.GetUID());
        auto lod = block_uid.GetLOD();
        aq.tasks[lod].emplace_back([this, block_uid = block_uid, buffer = std::move(buffer)]() mutable {
            volume->ReadBlock(block_uid, *buffer);
            buffer.SetUID(block_uid.ToUnifiedRescUID());
            buffer.ConvertWriteToReadLock();
            _AQ_AppendTransferTask(std::move(buffer));
        });
    }

    void _AQ_AppendTransferTask(Handle<CUDAHostBuffer> buffer){
        auto t = aq.async_transfer_queue.create_task([this, buffer = std::move(buffer)]() mutable {
            auto block_uid = BlockUID(buffer.GetUID());
            if(aq.cur_block_infos_mp.count(block_uid)){
                gpu_vtex_mgr_ref->UploadBlockToGPUTex(buffer, aq.cur_block_infos_mp.at(block_uid).second);
            }
            //上传到显存后，不会再使用，释放读锁
            buffer.ReleaseReadLock();
        });
        aq.async_transfer_queue.submit(t);
    }

    struct{
        Float3 lod0_block_length_space;
        UInt3 lod0_block_dim;
        BoundingBox3D volume_bound;
        int max_lod;
        Mat4 camera_proj_view;
        Float3 camera_pos;
        LevelOfDist lod;
    };

    UnifiedRescUID uid;

    std::mutex g_mtx;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::RTVolRenderer);
    }
};

RTVolumeRenderer::RTVolumeRenderer(const RTVolumeRenderer::RTVolumeRendererCreateInfo &info)
{
    _ = std::make_unique<RTVolumeRendererPrivate>();

    _->host_mem_mgr_ref = info.host_mem_mgr;
    _->gpu_mem_mgr_ref = info.gpu_mem_mgr;
    _->async = info.async;
    if(info.use_shared_host_mem)
        _->fixed_host_mem_mgr_ref = info.shared_fixed_host_mem_mgr_ref;
    else{
        //create fixed_host_mem_mgr until BindGridVolume

    }

    _->AQ_Init();
}

RTVolumeRenderer::~RTVolumeRenderer()
{

}

void RTVolumeRenderer::Lock()
{

}

void RTVolumeRenderer::UnLock()
{

}

UnifiedRescUID RTVolumeRenderer::GetUID() const
{
    return 0;
}

void RTVolumeRenderer::BindGridVolume(Handle<GridVolume>)
{
    // check if need create fixed_host_mem_mgr
    if(!_->fixed_host_mem_mgr_ref.IsValid()){


    }

    // check volume block length is same with fixed_host_mem_mgr
    {

    }

    // bind and set volume infos
    {

    }
}

void RTVolumeRenderer::SetRenderParams(const RenderParams &)
{

}

void RTVolumeRenderer::SetPerFrameParams(const PerFrameParams &)
{

}

void RTVolumeRenderer::SetRenderMode(bool async)
{

}

void RTVolumeRenderer::Render(Handle<FrameBuffer> frame)
{
    // get camera view frustum from per-frame-params

    Frustum camera_view_frustum;

    // compute current intersect blocks
    auto& intersect_blocks = _->intersect_blocks; intersect_blocks.clear();

    ComputeIntersectedBlocksWithViewFrustum(intersect_blocks,
                                            _->lod0_block_length_space,
                                            _->lod0_block_dim,
                                            _->volume_bound,
                                            camera_view_frustum,
                                            [max_lod = _->max_lod,
                                            camera_pos = _->camera_pos,
                                            this](const BoundingBox3D& box){
                                                auto center = (box.low + box.high) * 0.5f;
                                                float dist = (center - camera_pos).length();
                                                for(int i = 0; i <= max_lod; ++i){
                                                    if(dist < _->lod.LOD[i])
                                                        return i;
                                                }
                                                return max_lod;
                                            });


    // query from gpu page table
    auto& block_infos = _->block_infos; block_infos.clear();
    _->gpu_pt_mgr_ref->GetAndLock(intersect_blocks, block_infos);


    // add missed blocks into async-loading-queue
    _->AQ_Update(block_infos);

    _->AQ_Commit();

    // start render kernel

    // post-process

}

VISER_END