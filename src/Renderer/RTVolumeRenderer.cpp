#undef UTIL_ENABLE_OPENGL

#include <Core/Renderer.hpp>
#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <unordered_set>
#include <Core/HashPageTable.hpp>
#include "Common.hpp"
#include "../Common/helper_math.h"
#include <array>
#define Transform(t) t.x, t.y, t.z

VISER_BEGIN

using BlockUID = GridVolume::BlockUID;

static Int3 DefaultVTexShape{1024, 1024, 1024};

namespace{
    using namespace cuda;

    using HashTableItem = GPUPageTableMgr::PageTableItem;
    constexpr int HashTableSize = G_HashTableSize;
    constexpr int MaxHashTableQueryIdx = 1024;
    constexpr int MaxLodLevels = LevelOfDist::MaxLevelCount;
    constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5f;
//#define INVALID_HASH_TABLE_KEY uint4{0xffffu, 0xffffu, 0xffffu, 0xffu}


    CUB_GPU float gamma(int n) {
        return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
    }

    struct HashTable{
        HashTableItem hash_table[HashTableSize];
    };

    struct CUDAVolumeParams
    {
        AABB bound;
        float3 voxel_dim;
        float3 voxel_space;
        uint32_t block_length;
        uint32_t padding;
        uint32_t block_size;
    };

    struct CUDARenderParams{
        float lod_policy[MaxLodLevels];
        float ray_step;
        float max_ray_dist;//中心的最远距离
        float3 inv_tex_shape;
        bool use_2d_tf;
        int2 mpi_node_offset;
        int2 world_shape;
        bool gamma_correction;
        bool output_depth;
    };

    struct CUDAPerFrameParams{
        float3 cam_pos;
        float fov;
        float3 cam_dir;
        int frame_width;
        float3 cam_right;
        int frame_height;
        float3 cam_up;
        float frame_w_over_h;
        int debug_mode;
    };

    struct CUDAPageTable{
        CUDABufferView1D<HashTableItem> table;
    };

    struct CUDAFrameBuffer{
        CUDABufferView2D<uint32_t> color;
        CUDABufferView2D<float> depth;
        cudaSurfaceObject_t _color;
        cudaSurfaceObject_t _depth;
    };

    struct RTVolumeRenderKernelParams{
        CUDAVolumeParams cu_volume;
        CUDARenderParams cu_render_params;
        CUDAPerFrameParams cu_per_frame_params;
        CUDAPageTable cu_page_table;
        cudaTextureObject_t cu_vtex[MaxCUDATextureCountPerGPU];
        cudaTextureObject_t cu_tf_tex;
        cudaTextureObject_t cu_2d_tf_tex;
        CUDAFrameBuffer framebuffer;
    };

    struct RayCastResult{
        float4 color;
        float depth;
    };
    struct VirtualSamplingResult{
        uint32_t flag;
        float scalar;
    };

#define INVALID_VALUE uint4{0, 0, 0, 0}

    CUB_GPU uint32_t GetHashValue(const uint4& key){
        auto p = reinterpret_cast<const uint32_t*>(&key);
        uint32_t v = p[0];
        for(int i = 1; i < 4; i++){
            v = v ^ (p[i] + 0x9e3779b9 + (v << 6) + (v >> 2));
        }
        return v;
    }

    CUB_GPU uint4 Query(const uint4& key, uint4 hash_table[][2]){
        uint32_t hash_v = GetHashValue(key);
        auto pos = hash_v % HashTableSize;
        int i = 0;
        bool positive = false;
//        constexpr uint4 INVALID_HASH_TABLE_KEY = uint4{0xffffu, 0xffffu, 0xffffu, 0xffu};
        while(i < MaxHashTableQueryIdx){
            int ii = i * i;
            pos += positive ? ii : -ii;
            pos %= HashTableSize;
            if(hash_table[pos][0] == key){
                return hash_table[pos][1];
            }
            if(hash_table[pos][0].w == 0xffu){
                return INVALID_VALUE;
            }
//            if(i > 0){
//                printf("multi times query: %d\n", i * 2 + positive);
//            }
//            printf("key not find: %d %d %d %d, i : %d, positive : %d\n", key.x, key.y, key.z, key.w,
//                   i, int(positive));
            if(positive)
                ++i;
            positive = !positive;
        }
        return INVALID_VALUE;
    }

    CUB_GPU inline float3 CalcVirtualSamplingPos(const RTVolumeRenderKernelParams & params, const float3& pos){
        return (pos - params.cu_volume.bound.low)/(params.cu_volume.bound.high - params.cu_volume.bound.low);
    }

    CUB_GPU float CalcDistToNearestBlockCenter(const RTVolumeRenderKernelParams & params, const float3& pos){
        float3 offset_in_volume = CalcVirtualSamplingPos(params, pos);
        offset_in_volume *= params.cu_volume.voxel_dim;
        uint3 voxel_coord = make_uint3(offset_in_volume.x, offset_in_volume.y, offset_in_volume.z);
        voxel_coord /= params.cu_volume.block_length;
        voxel_coord = voxel_coord * params.cu_volume.block_length + params.cu_volume.block_length / 2;
        offset_in_volume =  make_float3(voxel_coord) / params.cu_volume.voxel_dim;
        float3 center_pos = offset_in_volume * (params.cu_volume.bound.high - params.cu_volume.bound.low) + params.cu_volume.bound.low;
        return length(params.cu_per_frame_params.cam_pos - center_pos);
    }

    CUB_GPU uint32_t ComputeLod(const RTVolumeRenderKernelParams & params, const float3& pos){
        float dist = CalcDistToNearestBlockCenter(params, pos);
        for(uint32_t i = 0; i < MaxLodLevels; ++i){
            if(dist < params.cu_render_params.lod_policy[i])
                return i;
        }
        return MaxLodLevels - 1;
    }

    CUB_GPU VirtualSamplingResult VirtualSampling(const RTVolumeRenderKernelParams & params,
                                                  uint4 hash_table[][2],
                                                  float3 offset_in_volume, uint32_t sampling_lod){
        //to voxel coord
        float3 sampling_coord =  offset_in_volume * params.cu_volume.voxel_dim;
        uint3 voxel_coord = make_uint3(sampling_coord.x, sampling_coord.y, sampling_coord.z);
        uint32_t lod_block_length = params.cu_volume.block_length << sampling_lod;
        uint3 block_uid = voxel_coord / lod_block_length;
        float3 offset_in_block = ((voxel_coord - block_uid * lod_block_length) + fracf(sampling_coord)) / float(1 << sampling_lod);
        uint4 key = make_uint4(block_uid, sampling_lod);
        uint4 tex_coord = Query(key, hash_table);
        uint32_t tid = (tex_coord.w >> 16) & 0xffff;
        //            printf("sampling tid : %d, %d %d %d %d\n", tid, tex_coord.x, tex_coord.y, tex_coord.z, tex_coord.w & 0xffff);
        uint3 coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
        VirtualSamplingResult ret{0, 0};
        if((tex_coord.w & 0xffff) & TexCoordFlag_IsValid){
            //                printf("sampling tex\n");
            // valid
            float3 sampling_pos = (coord * params.cu_volume.block_size + offset_in_block + params.cu_volume.padding) * params.cu_render_params.inv_tex_shape;

            ret.scalar = tex3D<float>(params.cu_vtex[tid], sampling_pos.x, sampling_pos.y, sampling_pos.z);
            //                cudaSurfaceObject_t sur;
            //                int data;
            //                surf2Dread(&data, sur, 1, 1);
            //                printf("sampling pos %f %f %f, value %f",
            //                       sampling_pos.x, sampling_pos.y, sampling_pos.z,
            //                       ret.scalar);

            ret.flag = tex_coord.w & 0xffff;
        }
        else{
//            if(params.cu_per_frame_params.debug_mode == 1)
//                printf("block not find : %d %d %d %d, %d %d %d %d\n",key.x, key.y, key.z, key.w,
//                       tex_coord.x, tex_coord.y, tex_coord.z, tex_coord.w);
        }
        return ret;
    }

    CUB_GPU float4 ScalarToRGBA(const RTVolumeRenderKernelParams & params,
                                float scalar, uint32_t lod){
        //            if(scalar > 0.3f){
        //                return make_float4(1.f,0.5f,0.25f,0.6f);
        //            }
        //            else return make_float4(0.f);
        if(scalar < 0.3f) return make_float4(0.f);
        if(params.cu_per_frame_params.debug_mode == 3){
            return make_float4(scalar, scalar, scalar, 1.f);
        }
        auto color = tex3D<float4>(params.cu_tf_tex, scalar, 0.5f, 0.5f);
        //            if(scalar >= 1.f){
        //                printf("scalar 0 to rgba %f %f %f %f\n", color.x, color.y, color.z, color.w);
        //            }
        return color;
    }
    CUB_GPU float4 CalcShadingColor(const RTVolumeRenderKernelParams & params,
                                    uint4 hash_table[][2],
                                    const float4& color, const float3& pos,
                                    const float3& ray_dir,
                                    const float3& dt, uint32_t lod){
        //todo use texGrad???
        float3 N;
        float x1, x2;
        int missed = 0;
        float lod_t = 1 << lod;
        auto ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(lod_t, 0.f, 0.f)), lod);
        x1 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(-lod_t, 0.f, 0.f)), lod);
        x2 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        N.x = x1 - x2;

        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, lod_t, 0.f)), lod);
        x1 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, -lod_t, 0.f)), lod);
        x2 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        N.y = x1 - x2;

        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, 0.f, lod_t)), lod);
        x1 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, 0.f, -lod_t)), lod);
        x2 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        N.z = x1 - x2;
        if(missed == 6){
            N = -ray_dir;
        }
        else
            N = -normalize(N);
        float3 albedo = make_float3(color);
        float3 ambient = 0.05f * albedo;
        float3 diffuse = max(dot(N, -ray_dir), 0.f) * albedo;
        float3 view_dir = normalize(params.cu_per_frame_params.cam_pos - pos);
        float3 specular = powf(max(0.f, dot(N, view_dir)), 24.f) * make_float3(0.2f);
        return make_float4(ambient + diffuse + specular, color.w);
    }

    CUB_GPU RayCastResult RayCastVolume(const RTVolumeRenderKernelParams & params,
                                        uint4 hash_table[][2],
                                        Ray ray){
        // default color is black and depth is 1.0
        RayCastResult ret{make_float4(0), 1.f};
        auto [entry_t, exit_t] = cuda::RayIntersectAABB(params.cu_volume.bound, ray);
        // ray has no intersection with volume
        if(entry_t >= exit_t || exit_t <= 0.f)
            return ret;

        entry_t = max(0.f ,entry_t);
        if(params.cu_per_frame_params.debug_mode == 1){
            ret.color = make_float4(ray.o + ray.d * entry_t, 1.f);
            return ret;
        }
        else if(params.cu_per_frame_params.debug_mode == 2){
            ret.color = make_float4(ray.o + ray.d * exit_t, 1.f);
            return ret;
        }

        float entry_to_exit_dist = exit_t - entry_t;
        // 注意浮点数的有效位数只有6位
        float3 ray_cast_pos = params.cu_per_frame_params.cam_pos + entry_t * ray.d;
        float dt = params.cu_render_params.ray_step;
        float ray_cast_dist = 0.f;

        uint32_t prev_lod = ComputeLod(params, ray_cast_pos);
        uint32_t pre_lod_sampling_steps = 0;
        float3 prev_lod_samping_pos = ray_cast_pos;
        uint32_t steps = 0;
        float ray_max_cast_dist = min(params.cu_render_params.max_ray_dist, entry_to_exit_dist);
        //            printf("ray_step : %.5f, ray_max_cast_dist : %.5f\n", dt, ray_max_cast_dist);
        //            return ret;

        while(ray_cast_dist < ray_max_cast_dist){
            float3 cur_ray_pos = ray_cast_pos;// + 0.5f * dt * ray.d;
            uint32_t cur_lod = ComputeLod(params, cur_ray_pos);
            bool upgrade = false;
            if(cur_lod > prev_lod){
                upgrade = true;
                prev_lod = cur_lod;
                prev_lod_samping_pos = cur_ray_pos;
                pre_lod_sampling_steps = steps;
            }
            float3 sampling_pos = CalcVirtualSamplingPos(params, cur_ray_pos);

            auto [flag, scalar] = VirtualSampling(params, hash_table, sampling_pos, cur_lod);

            //                bool skip = (flag == 0) || (flag & TexCoordFlag_IsValid && flag & TexCoordFlag_IsBlack);
            //                if(skip){
            //                    ray_cast_pos = CalcSkipPos(params, ray_cast_pos, ray.d, cur_lod);
            //
            //                    continue;
            //                }

            // compute and accumulate color
            // always assert scalar = 0.f has no meanings
            if(scalar > 0.f){
                //                    printf("sampling scalar %.5f\n", scalar);
                float4 mapping_color = ScalarToRGBA(params, scalar, cur_lod);
                //                    printf("sampling scalar %.5f, color %f %f %f %f\n",scalar, mapping_color.x, mapping_color.y, mapping_color.z, mapping_color.w);
                // always assert alpha = 0.f has no contribution
                if(mapping_color.w > 0.f){
                    if(params.cu_per_frame_params.debug_mode != 4)
                        mapping_color = CalcShadingColor(params, hash_table, mapping_color, cur_ray_pos, ray.d,
                                                         params.cu_volume.voxel_space, cur_lod);
                    // accumulate color radiance
                    ret.color += mapping_color * make_float4(make_float3(mapping_color.w), 1.f) * (1.f - ret.color.w);
                    if(ret.color.w > 0.99f){
                        break;
                    }
                }
            }
            //                ray_cast_pos += dt * ray.d;
            if(upgrade){
                dt = (1 << cur_lod) * params.cu_render_params.ray_step;
            }
            ray_cast_pos = prev_lod_samping_pos + (++steps - pre_lod_sampling_steps) * ray.d * dt;
            ray_cast_dist += dt;
        }
        // view space depth
        // todo return proj depth
        ret.depth = ray_cast_dist * dot(ray.d, params.cu_per_frame_params.cam_dir);
        return ret;
    }


    CUB_GPU uint32_t Float4ToUInt(const float4& f){
        uint32_t ret = (uint32_t)(saturate(f.x) * 255u)
                       | ((uint32_t)(saturate(f.y) * 255u) << 8)
                       | ((uint32_t)(saturate(f.z) * 255u) << 16)
                       | ((uint32_t)(saturate(f.w) * 255u) << 24);
        return ret;
    }
    CUB_GPU float3 PostProcessing(const float4& color){
        float3 ret;
        ret.x = pow(color.x, 1.f / 2.2f);
        ret.y = pow(color.y, 1.f / 2.2f);
        ret.z = pow(color.z, 1.f / 2.2f);
        return ret;
    }

    CUB_KERNEL void RTVolumeRenderKernel(RTVolumeRenderKernelParams params){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x >= params.cu_per_frame_params.frame_width || y >= params.cu_per_frame_params.frame_height)
            return;

        const unsigned int thread_count = blockDim.x * blockDim.y;
        const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
        const unsigned int load_count = (HashTableSize + thread_count - 1) / thread_count;
        const unsigned int thread_beg = thread_idx * load_count;
        const unsigned int thread_end = min(thread_beg + load_count, HashTableSize);

        __shared__ uint4 hash_table[HashTableSize][2];

        auto& table = params.cu_page_table.table;
        for(int i = 0; i < HashTableSize; ++i){
            auto& block_uid = table.at(i).first;
            auto& tex_coord = table.at(i).second;
            hash_table[i][0] = uint4{block_uid.x, block_uid.y, block_uid.z, block_uid.w};
            hash_table[i][1] = uint4{tex_coord.sx, tex_coord.sy, tex_coord.sz,
                                     ((uint32_t)tex_coord.tid << 16) | tex_coord.flag};
        }

        __syncthreads();
//        {
//            float4 color = {1.f, 0.f, 0.f, 1.f};
//            surf2Dwrite(Float4ToUInt(color), params.framebuffer._color, x * sizeof(uint32_t), y);
//            return;
//        }
        Ray ray;
        ray.o = params.cu_per_frame_params.cam_pos;
        float scale = tanf(0.5f * params.cu_per_frame_params.fov);
        float ix = (x + params.cu_render_params.mpi_node_offset.x + 0.5f) /
                       (params.cu_per_frame_params.frame_width * params.cu_render_params.world_shape.x) - 0.5f;
        float iy = (y + params.cu_render_params.mpi_node_offset.y + 0.5f) /
                       (params.cu_per_frame_params.frame_height * params.cu_render_params.world_shape.y) - 0.5f;
        ray.d = params.cu_per_frame_params.cam_dir + params.cu_per_frame_params.cam_up * scale * iy
                + params.cu_per_frame_params.cam_right * scale * ix * params.cu_per_frame_params.frame_w_over_h;


        auto [color, depth] = RayCastVolume(params, hash_table, ray);

        if(params.cu_render_params.gamma_correction){
            color = make_float4(PostProcessing(color), color.w);
            // exposure gamma color-grading tone-mapping...
        }
        //            y = params.cu_per_frame_params.frame_height - 1 - y;
        if(params.framebuffer.color.data())
            params.framebuffer.color.at(x, y) = Float4ToUInt(color);
        else{

            surf2Dwrite(Float4ToUInt(color), params.framebuffer._color, x * sizeof(uint32_t), y);
//            printf("using surface to output color\n");
        }

        if(params.cu_render_params.output_depth){
            if(params.framebuffer.depth.data())
                params.framebuffer.depth.at(x, y) = depth;
            else
                surf2Dwrite(depth, params.framebuffer._depth, x, y);
        }
    }

}

class RTVolumeRendererPrivate{
  public:
    struct{
        CUDAContext ctx;
        CUDAStream stream;
        RTVolumeRenderKernelParams kernel_params;
    };


    struct{
        Handle<CUDAHostBuffer> tf1d;
        Handle<CUDAHostBuffer> tf2d;
        Handle<CUDATexture> cu_tf_tex;
        Handle<CUDATexture> cu_2d_tf_tex;
        int tf_dim = 0;
    };

    void GenTFTex(int dim){
        if(tf_dim == dim) return;
        tf_dim = dim;
        tf1d.Destroy();
        tf2d.Destroy();
        cu_tf_tex.Destroy();
        cu_2d_tf_tex.Destroy();

        size_t bytes_1d = sizeof(typename TransferFunc::Value) * dim;
        tf1d = NewHandle<CUDAHostBuffer>(ResourceType::Buffer, bytes_1d, cub::e_cu_host, ctx);

        size_t bytes_2d = sizeof(typename TransferFunc::Value) * dim * dim;
        tf2d = NewHandle<CUDAHostBuffer>(ResourceType::Buffer, bytes_2d, cub::e_cu_host, ctx);

        cub::texture_resc_info resc_info{cub::e_float, 4, {256, 1, 1}};
        cub::texture_view_info view_info; view_info.read = cub::e_raw;
        cu_tf_tex = NewHandle<CUDATexture>(ResourceType::Buffer, resc_info, view_info, ctx);

        resc_info.extent = {256, 256, 1};
        cu_2d_tf_tex = NewHandle<CUDATexture>(ResourceType::Buffer, resc_info, view_info, ctx);
    }

    struct{
        Ref<HostMemMgr> host_mem_mgr_ref;
        Ref<GPUMemMgr> gpu_mem_mgr_ref;
        Ref<GPUVTexMgr> gpu_vtex_mgr_ref;
        Ref<GPUPageTableMgr> gpu_pt_mgr_ref;
        Ref<FixedHostMemMgr> fixed_host_mem_mgr_ref;

        Handle<GridVolume> volume;
    };

    struct{
        float render_space_ratio;
        bool use_shared_host_mem;
        size_t fixed_host_mem_bytes;
        size_t vtex_cnt;
        Int3 vtex_shape;
    };

    struct{
        Float3 lod0_block_length_space;
        UInt3 lod0_block_dim;
        BoundingBox3D volume_bound;
        int max_lod;
        Float3 camera_pos;
        LevelOfDist lod;
        Mat4 proj_view;
        int world_rows;
        int world_cols;
        int node_x_idx;
        int node_y_idx;
    };
    // data sts for loading blocks
    struct{
        bool async;
        std::vector<BlockUID> intersect_blocks;
//        std::vector<GPUPageTableMgr::PageTableItem> block_infos;

    };

    struct{
        vutil::thread_group_t async_loading_queue;
        vutil::thread_group_t async_decoding_queue;
        vutil::thread_group_t async_transfer_queue;
        std::mutex decoding_queue_mtx;
        std::mutex transfer_queue_mtx;
//        std::vector<Handle<CUDAHostBuffer>> cached_host_blocks;
//        std::vector<Handle<CUDAHostBuffer>> missed_host_blocks;
//        std::unordered_set<BlockUID> cur_blocks_st;
        std::unordered_map<BlockUID, GPUPageTableMgr::PageTableItem> cur_block_infos_mp;
        std::mutex block_infos_mtx;
        std::map<int, std::vector<std::function<void()>>> tasks;
    }aq;

    void AQ_Init(){
        aq.async_loading_queue.start(6);
        aq.async_decoding_queue.start(3);
        aq.async_transfer_queue.start(2);
    }

    void AQ_Update(const std::vector<BlockUID>& intersect_blocks){
        std::vector<BlockUID> missed_blocks;
        std::vector<GPUPageTableMgr::PageTableItem> block_infos;
//        aq.cur_block_infos_mp.clear();
        {
            std::lock_guard<std::mutex> lk(aq.block_infos_mtx);

            gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::GetAndLock, intersect_blocks, block_infos);

            std::unordered_map<BlockUID, GPUPageTableMgr::PageTableItem> dummy;
            for (auto &[block_uid, tex_coord] : block_infos)
            {
                if (!tex_coord.Missed())
                    continue;
                dummy[block_uid] = {block_uid, tex_coord};
                if (aq.cur_block_infos_mp.count(block_uid) == 0)
                {
                    missed_blocks.emplace_back(block_uid);
                }
            }
            aq.cur_block_infos_mp = std::move(dummy);
        }

        auto task = aq.async_loading_queue.create_task([&, missed_blocks = std::move(missed_blocks)]{
            std::vector<Handle<CUDAHostBuffer>> buffers;
//            std::lock_guard<std::mutex> lk(aq.block_infos_mtx);
            fixed_host_mem_mgr_ref._get_ptr()->Lock();
            for(auto& block_uid : missed_blocks){
                LOG_DEBUG("start loading block: {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
                auto hd = fixed_host_mem_mgr_ref._get_ptr()->GetBlock(block_uid.ToUnifiedRescUID());
//                if(!hd.IsValid()){
////                    LOG_DEBUG("Invalid erase block: {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
//                    aq.cur_block_infos_mp.erase(block_uid);
//
//                    continue;
//                }

                auto& b = buffers.emplace_back(hd.SetUID(block_uid.ToUnifiedRescUID()));

                LOG_DEBUG("loading block: {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
                if(!b.IsLocked()){
                    LOG_ERROR("no locked block: {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
                }
            }
            fixed_host_mem_mgr_ref._get_ptr()->UnLock();
            _AQ_AppendTask(std::move(buffers));
        });
        aq.async_loading_queue.submit(task);
    }


    void _AQ_AppendTask(std::vector<Handle<CUDAHostBuffer>> buffers){
        // just consider cpu decoding now
        // if read lock, then transfer
        // if write lock, then loading and transfer
        for(auto& buffer : buffers){
            if(buffer.IsWriteLocked()){
                _AQ_AppendDecodingTask(std::move(buffer));
            }
            else if(buffer.IsReadLocked()){
                _AQ_AppendTransferTask(std::move(buffer));
            }
            else{
                assert(false);
            }
        }
    }

    void AQ_Commit(bool updatePT = true){
        // submit loading tasks
        std::map<int, vutil::task_group_handle_t> task_groups;
        std::vector<int> lods;
        {
            std::lock_guard<std::mutex> lk(aq.decoding_queue_mtx);
            for (auto &[lod, tasks] : aq.tasks)
            {
                lods.emplace_back(lod);
                auto task_group = aq.async_decoding_queue.create_task();
                for (auto &task : tasks)
                {
                    task_group->enqueue_task(std::move(task));
                }
                task_groups[lod] = std::move(task_group);
            }
            aq.tasks.clear();
        }
        int lod_count = lods.size();
        for (int lod = 0; lod < lod_count - 1; ++lod)
        {
            int first = lods[lod], second = lods[lod + 1];
            aq.async_decoding_queue.add_dependency(*task_groups[second], *task_groups[first]);
        }
        for (auto &[lod, task_group] : task_groups)
        {
            aq.async_decoding_queue.submit(task_group);
        }

        // wait all transfer tasks finished
        aq.async_transfer_queue.wait_idle();
        if(updatePT){
            //update page table
            kernel_params.cu_page_table.table = gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::GetPageTable, true).
                                                GetHandle()->view_1d<HashTableItem>(HashTableSize * sizeof(HashTableItem));
        }
    }

    void AQ_Wait(){
        auto all_is_ok = [&] {
            return aq.async_loading_queue.is_idle()
            && aq.async_decoding_queue.is_idle()
            && aq.async_transfer_queue.is_idle();
        };
        while(!all_is_ok())
            AQ_Commit(false);
        AQ_Commit(true);
    }

    void _AQ_AppendDecodingTask(Handle<CUDAHostBuffer> buffer){
        auto block_uid = BlockUID(buffer.GetUID());
        auto lod = block_uid.GetLOD();
        std::lock_guard<std::mutex> lk(aq.decoding_queue_mtx);
        aq.tasks[lod].emplace_back([this, block_uid = block_uid, buffer = std::move(buffer)]() mutable {
           //CPU decoding
           LOG_DEBUG("decoding start, {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
           {
               auto lk = volume.AutoLocker();
               volume->ReadBlock(block_uid, *buffer);
           }
           //            buffer.SetUID(block_uid.ToUnifiedRescUID());
            if(!buffer.IsWriteLocked()){
                LOG_ERROR("buffer not write locked");
            }
            buffer.ConvertWriteToReadLock();
            LOG_DEBUG("decoding finish, {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
            _AQ_AppendTransferTask(std::move(buffer));
        });
    }

    void _AQ_AppendTransferTask(Handle<CUDAHostBuffer> buffer){
        std::lock_guard<std::mutex> lk(aq.transfer_queue_mtx);
        auto t = aq.async_transfer_queue.create_task([this, buffer = std::move(buffer)]() mutable {
            auto block_uid = BlockUID(buffer.GetUID());
            LOG_DEBUG("transfer start, {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
            {

                std::lock_guard<std::mutex> lk(aq.block_infos_mtx);
                if (aq.cur_block_infos_mp.count(block_uid))
                {
                    auto dst = aq.cur_block_infos_mp.at(block_uid).second;
                    auto ret = gpu_vtex_mgr_ref.Invoke(&GPUVTexMgr::UploadBlockToGPUTex, buffer,
                                            dst);
                    if(ret){
                        aq.cur_block_infos_mp.erase(block_uid);
                        gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::Release, std::vector<BlockUID>{block_uid}, false);
                    }
                    else{
                        gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::Discard, std::vector<BlockUID>{block_uid});
                    }
                }
                else{
                    //release write lock for pt
                    gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::Discard, std::vector<BlockUID>{block_uid});
                }

            }
            //上传到显存后，不会再使用，释放读锁
            buffer.ReleaseReadLock();
            LOG_DEBUG("transfer finish, {} {} {} {}", block_uid.x, block_uid.y, block_uid.z, block_uid.GetLOD());
        });
        aq.async_transfer_queue.submit(t);
    }

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

    _->ctx = _->gpu_mem_mgr_ref._get_ptr()->_get_cuda_context();
    _->stream = cub::cu_stream::null(_->ctx);

    _->uid = _->GenRescUID();

    if(info.use_shared_host_mem){
        _->fixed_host_mem_mgr_ref = info.shared_fixed_host_mem_mgr_ref;
        LOG_DEBUG("ref is locked {}", _->fixed_host_mem_mgr_ref.IsThreadSafe());
    }
    else{
        //create fixed_host_mem_mgr until BindGridVolume

    }

    {
        _->fixed_host_mem_bytes = info.fixed_host_mem_bytes;
        _->vtex_cnt = info.vtex_cnt;
        _->vtex_shape = info.vtex_shape;
        //todo move to RenderParams
        _->render_space_ratio = info.render_space_ratio;
        _->use_shared_host_mem = info.use_shared_host_mem;
    }

    _->AQ_Init();
}

RTVolumeRenderer::~RTVolumeRenderer()
{

}

void RTVolumeRenderer::Lock()
{
    _->g_mtx.lock();
}

void RTVolumeRenderer::UnLock()
{
    _->g_mtx.unlock();
}

UnifiedRescUID RTVolumeRenderer::GetUID() const
{
    return _->uid;
}

void RTVolumeRenderer::BindGridVolume(Handle<GridVolume> volume)
{
    // check if re-create fixed_host_mem_mgr and gpu_vtex_mgr
    auto should_recreate_mem_mgr = [&]()->std::pair<bool, bool>{
        std::pair<bool,bool> ret = {false, false};
        if(!_->gpu_vtex_mgr_ref.IsValid())
            ret.second = true;
        if(_->volume.IsValid()){
            auto desc0 = _->volume->GetDesc();
            auto desc1 = volume->GetDesc();
            if (desc0.block_length + desc0.padding != desc1.block_length + desc1.padding)
                ret = {true, true};
        }
        else ret = {true, true};
        if(_->use_shared_host_mem) ret.first = false;
        else if(!_->fixed_host_mem_mgr_ref.IsValid()) ret.first = true;
        return ret;
    };
    auto [should_host, should_gpu] = should_recreate_mem_mgr();
    auto desc = volume->GetDesc();
    {
        size_t block_size = (desc.block_length + desc.padding * 2) * desc.bits_per_sample *
                            desc.samples_per_voxel / 8;
        block_size *= block_size * block_size;
        size_t block_num = _->fixed_host_mem_bytes / block_size;

        if(should_host){
            FixedHostMemMgr::FixedHostMemMgrCreateInfo fixed_info;
            fixed_info.host_mem_mgr = _->host_mem_mgr_ref;
            fixed_info.fixed_block_size = block_size;
            fixed_info.fixed_block_num = block_num;

            auto fixed_host_mem_uid = _->host_mem_mgr_ref.Invoke(&HostMemMgr::RegisterFixedHostMemMgr, fixed_info);
            _->fixed_host_mem_mgr_ref = _->host_mem_mgr_ref.Invoke(&HostMemMgr::GetFixedHostMemMgrRef, fixed_host_mem_uid).LockRef();
        }

        if(should_gpu){
            GPUVTexMgr::GPUVTexMgrCreateInfo vtex_info;
            vtex_info.gpu_mem_mgr = _->gpu_mem_mgr_ref;
            vtex_info.host_mem_mgr = _->host_mem_mgr_ref;
            vtex_info.vtex_count = _->vtex_cnt;
            vtex_info.vtex_shape = _->vtex_shape;
            vtex_info.bits_per_sample = desc.bits_per_sample;
            vtex_info.samples_per_channel = desc.samples_per_voxel;
            vtex_info.vtex_block_length = (desc.block_length + desc.padding * 2);
            vtex_info.is_float = desc.is_float;
            vtex_info.exclusive = true;

            auto vtex_uid = _->gpu_mem_mgr_ref.Invoke(&GPUMemMgr::RegisterGPUVTexMgr, vtex_info);
            _->gpu_vtex_mgr_ref = _->gpu_mem_mgr_ref._get_ptr()->GetGPUVTexMgrRef(vtex_uid).LockRef();
            _->gpu_pt_mgr_ref = _->gpu_vtex_mgr_ref._get_ptr()->GetGPUPageTableMgrRef().LockRef();
        }
    }


    {
        //bind vtex
        auto texes = _->gpu_vtex_mgr_ref.Invoke(&GPUVTexMgr::GetAllTextures);
        for(auto [unit, tex] : texes){
            _->kernel_params.cu_vtex[unit] = tex->_get_tex_handle();
        }

        // set here?
        _->kernel_params.cu_render_params.inv_tex_shape = float3{
            1.f / _->vtex_shape.x,
            1.f / _->vtex_shape.y,
            1.f / _->vtex_shape.z
        };

        //bind volume params
        _->volume = std::move(volume);

        _->kernel_params.cu_volume.block_length = desc.block_length;
        _->kernel_params.cu_volume.padding = desc.padding;
        _->kernel_params.cu_volume.block_size = desc.block_length + desc.padding * 2;
        _->kernel_params.cu_volume.voxel_dim = {(float)desc.shape.x,
                                                (float)desc.shape.y,
                                                (float)desc.shape.z};
        _->kernel_params.cu_volume.voxel_space = {_->render_space_ratio * desc.voxel_space.x,
                                                  _->render_space_ratio * desc.voxel_space.y,
                                                  _->render_space_ratio * desc.voxel_space.z};
        _->kernel_params.cu_volume.bound = {
            {0.f, 0.f, 0.f},
            _->kernel_params.cu_volume.voxel_space * _->kernel_params.cu_volume.voxel_dim};

        {
            _->lod0_block_dim = desc.blocked_dim;
            _->lod0_block_length_space = (float)desc.block_length * desc.voxel_space;
            _->volume_bound = {
                Float3(0.f),
                Float3(desc.shape.x * desc.voxel_space.x,
                       desc.shape.y * desc.voxel_space.y,
                       desc.shape.z * desc.voxel_space.z)
            };
            _->max_lod = _->volume->GetMaxLOD();
        }
    }
}

void RTVolumeRenderer::SetRenderParams(const RenderParams &render_params)
{
    if(render_params.light.updated){

    }
    if(render_params.lod.updated){
        std::memcpy(_->kernel_params.cu_render_params.lod_policy, render_params.lod.leve_of_dist.LOD, sizeof(float) * MaxLodLevels);
        _->lod = render_params.lod.leve_of_dist;
    }
    if(render_params.tf.updated){
        _->GenTFTex(render_params.tf.dim);
        render_params.tf.tf_pts.Gen1DTF(_->tf1d, render_params.tf.dim);
        render_params.tf.tf_pts.Gen2DTF(_->tf1d, _->tf2d, render_params.tf.dim);
        // transfer to texture
        cub::memory_transfer_info info;
        info.width_bytes = sizeof(TransferFunc::Value) * render_params.tf.dim;
        info.height = info.depth = 1;
//        VISER_WHEN_DEBUG({
//            auto v = _->tf1d->view_1d<float4>(256);
//            for (int i = 0; i < 256; i++)
//            {
//                std::cout << i << " : " << v.at(i).x << " " << v.at(i).y << " " << v.at(i).z << " " << v.at(i).w
//                          << std::endl;
//            }
//        });
        cub::cu_memory_transfer(*_->tf1d, *_->cu_tf_tex, info).launch(_->stream);
        info.height = render_params.tf.dim;
        cub::cu_memory_transfer(*_->tf2d, *_->cu_2d_tf_tex, info).launch(_->stream);

        _->kernel_params.cu_tf_tex = _->cu_tf_tex->_get_tex_handle();
        _->kernel_params.cu_2d_tf_tex = _->cu_2d_tf_tex->_get_tex_handle();
    }
    if(render_params.distrib.updated){
        _->kernel_params.cu_render_params.mpi_node_offset = {render_params.distrib.node_x_offset,
                                                              render_params.distrib.node_y_offset};
        _->kernel_params.cu_render_params.world_shape = {render_params.distrib.world_col_count,
                                                         render_params.distrib.world_row_count};
        _->world_rows = render_params.distrib.world_row_count;
        _->world_cols = render_params.distrib.world_col_count;
        _->node_x_idx = render_params.distrib.node_x_index;
        _->node_y_idx = render_params.distrib.node_y_index;
    }
    if(render_params.raycast.updated){
        _->kernel_params.cu_render_params.ray_step = render_params.raycast.ray_step;
        _->kernel_params.cu_render_params.max_ray_dist = render_params.raycast.max_ray_dist;
    }
    if(render_params.other.updated){
        //todo
//        _->kernel_params.cu_render_params.inv_tex_shape = float3{
//            render_params.other.inv_tex_shape.x,
//            render_params.other.inv_tex_shape.y,
//            render_params.other.inv_tex_shape.z
//        };
        _->kernel_params.cu_render_params.use_2d_tf = render_params.other.use_2d_tf;
        _->kernel_params.cu_render_params.gamma_correction = render_params.other.gamma_correct;
        _->kernel_params.cu_render_params.output_depth = render_params.other.output_depth;
    }
}

void RTVolumeRenderer::SetPerFrameParams(const PerFrameParams &per_frame_params)
{
    _->kernel_params.cu_per_frame_params.cam_pos = { Transform(per_frame_params.cam_pos) };
    _->kernel_params.cu_per_frame_params.fov = per_frame_params.fov;
    _->kernel_params.cu_per_frame_params.cam_dir = { Transform(per_frame_params.cam_dir) };
    _->kernel_params.cu_per_frame_params.frame_width = per_frame_params.frame_width;
    _->kernel_params.cu_per_frame_params.cam_right = { Transform(per_frame_params.cam_right) };
    _->kernel_params.cu_per_frame_params.frame_height = per_frame_params.frame_height;
    _->kernel_params.cu_per_frame_params.cam_up = { Transform(per_frame_params.cam_up) };
    _->kernel_params.cu_per_frame_params.frame_w_over_h = per_frame_params.frame_w_over_h;
//    _->kernel_params.cu_per_frame_params.debug_mode = per_frame_params.debug_mode;

    _->camera_pos = per_frame_params.cam_pos;
    _->proj_view = per_frame_params.proj_view;
}

void RTVolumeRenderer::SetRenderMode(bool async)
{
    _->async = async;
}

void RTVolumeRenderer::Render(Handle<FrameBuffer> frame)
{
    // get camera view frustum from per-frame-params

    Frustum camera_view_frustum;
    {
        // calc world camera corners
        ExtractFrustumFromMatrix(_->proj_view, camera_view_frustum);
        auto _calc = [](const Float3& a, const Float3& b, int n, int i){
            std::pair<Float3, Float3> ret;
            ret.first = a + (b - a) * static_cast<float>(i) / static_cast<float>(n);
            ret.second = a + (b - a) * static_cast<float>(i + 1) / static_cast<float>(n);
            return ret;
        };
//        auto& o_corners = camera_view_frustum.frustum_corners;
//        auto [_n_lb, _n_rb] = _calc(o_corners[0], o_corners[1], _->world_cols, _->node_x_idx);
//        auto [_n_lt, _n_rt] = _calc(o_corners[2], o_corners[3], _->world_cols, _->node_x_idx);
//        auto [n_lb, n_lt]   = _calc(_n_lb, _n_lt, _->world_rows, _->node_y_idx);
//        auto [n_rb, n_rt]   = _calc(_n_rb, _n_rt, _->world_rows, _->node_y_idx);
//        auto [_f_lb, _f_rb] = _calc(o_corners[4], o_corners[5], _->world_cols, _->node_x_idx);
//        auto [_f_lt, _f_rt] = _calc(o_corners[6], o_corners[7], _->world_cols, _->node_x_idx);
//        auto [f_lb, f_lt]   = _calc(_f_lb, _f_lt, _->world_rows, _->node_y_idx);
//        auto [f_rb, f_rt]   = _calc(_f_rb, _f_rt, _->world_rows, _->node_y_idx);
//        o_corners[0] = n_lb, o_corners[1] = n_rb, o_corners[2] = n_lt, o_corners[3] = n_rt;
//        o_corners[4] = f_lb, o_corners[5] = f_rb, o_corners[6] = f_lt, o_corners[7] = f_rt;
//        std::array<Float3,8> corners;
//        for(int i = 0; i < 8; i++) corners[i] = camera_view_frustum.frustum_corners[i];
//        vutil::extract_frustum_from_corners(corners, camera_view_frustum);
    }

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
//    auto& block_infos = _->block_infos; block_infos.clear();
//    _->gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::GetAndLock, intersect_blocks, block_infos);


    // add missed blocks into async-loading-queue
    _->AQ_Update(intersect_blocks);

    _->AQ_Commit();

    if(!_->async){
        _->AQ_Wait();
    }
    // start render kernel
    {
        const dim3 tile = {16u, 16u, 1u};
        _->kernel_params.framebuffer.color = frame->color;
        _->kernel_params.framebuffer.depth = frame->depth;
        _->kernel_params.framebuffer._color = frame->_color->_get_handle();
        if(_->kernel_params.cu_render_params.output_depth)
            _->kernel_params.framebuffer._depth = frame->_depth->_get_handle();

        cub::cu_kernel_launch_info launch_info;
        launch_info.shared_mem_bytes = 0;//CRTVolumeRendererPrivate::shared_mem_size;
        launch_info.block_dim = tile;
        launch_info.grid_dim = {(frame->frame_width + tile.x - 1) / tile.x,
                                (frame->frame_height + tile.y - 1) / tile.y, 1};
        void* params[] = {&_->kernel_params};

        auto render_task = cub::cu_kernel::pending(launch_info, &RTVolumeRenderKernel, params);
        try{
            render_task.launch(_->stream).check_error_on_throw();
        }
        catch (const std::exception& err) {
            LOG_ERROR("RTVolumeRenderer render frame failed : {}", err.what());
        }

    }

    // post-process
    _->gpu_pt_mgr_ref.Invoke(&GPUPageTableMgr::Release, intersect_blocks, true);



}

VISER_END