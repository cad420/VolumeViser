#undef UTIL_ENABLE_OPENGL

#include <Core/Renderer.hpp>
#include "Common.hpp"
#include "../Common/helper_math.h"

VISER_BEGIN

#define Transform(t) t.x, t.y, t.z

    namespace {
        using HashTableItem = GPUPageTableMgr::PageTableItem;
        static constexpr int HashTableSize = 1024;
        static constexpr int MaxLodLevels = LevelOfDist::MaxLevelCount;
        static constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5f;

        CUB_GPU float gamma(int n) {
            return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
        }

        struct HashTable{
            HashTableItem hash_table[HashTableSize];
        };

        using namespace cuda;

        //...
        //体数据的包围盒可以是经过线性变换的，这里就用AABB，即只考虑平移和缩放，旋转由相机得到
        struct CUDAVolume{
            AABB bound;
            float3 voxel_dim;
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
        };
        //64kb


        struct CUDAPerFrameParams{
            float3 cam_pos;
            float fov;
            float3 cam_dir;
            int frame_width;
            float3 cam_right;
            int frame_height;
            float3 cam_up;
            float frame_w_over_h;
        };

        struct CUDAPageTable{
            CUDABufferView1D<HashTableItem> table;
        };

        struct CUDAFrameBuffer{
            CUDABufferView2D<uint32_t> color;
            CUDABufferView2D<float> depth;
        };

        struct CTRVolumeRenderKernelParams{
            CUDAVolume cu_volume;
            CUDARenderParams cu_render_params;
            CUDAPerFrameParams cu_per_frame_params;
            CUDAPageTable cu_page_table;
            cudaTextureObject_t cu_vtex[MaxCUDATextureCountPerGPU];
            cudaTextureObject_t cu_tf_tex;
            cudaTextureObject_t cu_2d_tf_tex;
            CUDAFrameBuffer framebuffer;
        };
#ifdef UNIQUE_RENDER_PROCESS
        //todo
        __constant__ CUDAVolume cu_volume;
        __constant__ CUDARenderParams cu_render_params;
        __constant__ CUDAPerFrameParams cu_per_frame_params;
        __constant__ CUDAHashTable cu_hash_table;
        __constant__ CUDATexture cu_vtex[MaxCUDATextureCountPerGPU];
#endif

        CUB_GPU uint32_t GetHashValue(const uint4& key){
            auto p = reinterpret_cast<const uint32_t*>(&key);
            uint32_t v = p[0];
            for(int i = 1; i < 4; i++){
                v = v ^ (p[i] + 0x9e3779b9 + (v << 6) + (v >> 2));
            }
            return v;
        }

#define INVALID_VALUE uint4{0, 0, 0, 0}

        CUB_GPU uint4 Query(const uint4& key, uint4 hash_table[][2]){
            uint32_t hash_v = GetHashValue(key);
            auto pos = hash_v % HashTableSize;
            int i = 0;
            bool positive = false;
            while(i < HashTableSize){
                int ii = i * i;
                pos += positive ? ii : -ii;
                pos %= HashTableSize;
                if(hash_table[pos][0] == key){
                    return hash_table[pos][1];
                }
                if(positive)
                    ++i;
                positive = !positive;
            }
            return INVALID_VALUE;
        }

        struct RayCastResult{
            float4 color;
            float depth;
        };
        struct VirtualSamplingResult{
            uint32_t flag;
            float scalar;
        };
        /**
         * @brief 不检查传入的坐标是否在包围盒內部，因此返回的值可能在0~1之外
         * @param pos world space coord
         * @return offset coord in volume aabb
         */
        CUB_GPU inline float3 CalcVirtualSamplingPos(const CTRVolumeRenderKernelParams& params, const float3& pos){
            return (pos - params.cu_volume.bound.low)/(params.cu_volume.bound.high - params.cu_volume.bound.low);
        }
        /**
         * @param offset_in_volume 采样点在AABB中的offset, 0 ~ 1
         */
        CUB_GPU VirtualSamplingResult VirtualSampling(const CTRVolumeRenderKernelParams& params,
                                                      uint4 hash_table[][2],
                                                      float3 offset_in_volume, uint32_t sampling_lod){
            //to voxel coord
            float3 sampling_coord =  offset_in_volume * params.cu_volume.voxel_dim;
            uint3 voxel_coord = make_uint3(sampling_coord.x, sampling_coord.y, sampling_coord.z);
            uint32_t lod_block_length = params.cu_volume.block_length << sampling_lod;
            uint3 block_uid = voxel_coord / lod_block_length;
            float3 offset_in_block = sampling_coord - block_uid * lod_block_length;
            uint4 key = make_uint4(block_uid, sampling_lod);
            uint4 tex_coord = Query(key, hash_table);
            uint32_t tid = (tex_coord.w >> 16) & 0xffff;
            uint3 coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
            VirtualSamplingResult ret{0, 0};
            if((tex_coord.w & 0xffff) & TexCoordFlag_IsValid){
                // valid
                float3 sampling_pos = (coord * params.cu_volume.block_size + offset_in_block + params.cu_volume.padding) * params.cu_render_params.inv_tex_shape;

                ret.scalar = tex3D<float>(params.cu_vtex[tid], sampling_pos.x, sampling_pos.y, sampling_pos.z);
                ret.flag = tex_coord.w & 0xffff;
            }
            return ret;
        }
        /**
         * @brief 总是使用lod0的block进行计算
         */
        CUB_GPU float CalcDistToNearestBlockCenter(const CTRVolumeRenderKernelParams& params, const float3& pos){
            float3 offset_in_volume = CalcVirtualSamplingPos(params, pos);
            offset_in_volume *= params.cu_volume.voxel_dim;
            uint3 voxel_coord = make_uint3(offset_in_volume.x, offset_in_volume.y, offset_in_volume.z);
            voxel_coord /= params.cu_volume.block_length;
            voxel_coord = voxel_coord * params.cu_volume.block_length + params.cu_volume.block_length / 2;
            offset_in_volume =  make_float3(voxel_coord) / params.cu_volume.voxel_dim;
            float3 center_pos = offset_in_volume * (params.cu_volume.bound.high - params.cu_volume.bound.low) + params.cu_volume.bound.low;
            return length(pos - center_pos);
        }
        /**
         * @param pos world space coord
         */
        CUB_GPU uint32_t ComputeLod(const CTRVolumeRenderKernelParams& params, const float3& pos){
            float dist = CalcDistToNearestBlockCenter(params, pos);
            for(uint32_t i = 0; i < MaxLodLevels; ++i){
                if(dist < params.cu_render_params.lod_policy[i])
                    return i;
            }
            return MaxLodLevels - 1;
        }

        CUB_GPU float3 CalcSkipPos(const CTRVolumeRenderKernelParams& params,
                                   const float3& pos, const float3& d, uint32_t lod){
            float3 low;
            float3 high;
            auto [_, exit_t] = cuda::RayIntersectAABB(low, high, Ray{pos, d});
            // robust bounds intersection
            exit_t *= 1.f + 2.f * gamma(3);
            return pos + exit_t * d;
        }
        CUB_GPU float4 ScalarToRGBA(const CTRVolumeRenderKernelParams& params,
                                    float scalar, uint32_t lod){
            auto color = tex1D<float4>(params.cu_tf_tex, scalar);
            return color;
        }
        CUB_GPU float4 CalcShadingColor(const CTRVolumeRenderKernelParams& params,
                                        const float4& color, const float3 pos, uint32_t lod){
            // todo
            return color;
        }
        CUB_GPU RayCastResult RayCastVolume(const CTRVolumeRenderKernelParams& params,
                                            uint4 hash_table[][2],
                                            Ray ray){
            // default color is black and depth is 1.0
            RayCastResult ret{make_float4(0), 1.f};
            auto [entry_t, exit_t] = cuda::RayIntersectAABB(params.cu_volume.bound, ray);
            // ray has no intersection with volume
            if(entry_t >= exit_t || exit_t <= 0.f)
                return ret;

            entry_t = max(0.f ,entry_t);
            float entry_to_exit_dist = exit_t - entry_t;
            // 注意浮点数的有效位数只有6位
            float3 ray_cast_pos = params.cu_per_frame_params.cam_pos + entry_t * ray.d;
            float dt = params.cu_render_params.ray_step;
            float ray_cast_dist = 0.f;

            uint32_t prev_lod;
            float ray_max_cast_dist = min(params.cu_render_params.max_ray_dist, entry_to_exit_dist);
            while(ray_cast_dist < ray_max_cast_dist){
                float3 sampling_pos = ray_cast_pos + 0.5f * dt * ray.d;

                uint32_t cur_lod = ComputeLod(params, sampling_pos);
                bool upgrade = false;
                if(cur_lod > prev_lod){
                    upgrade = true;
                    prev_lod = cur_lod;
                }
                sampling_pos = CalcVirtualSamplingPos(params, sampling_pos);

                auto [flag, scalar] = VirtualSampling(params, hash_table, sampling_pos, cur_lod);

                bool skip = (flag == 0) || (flag & TexCoordFlag_IsValid && flag & TexCoordFlag_IsBlack);
                if(skip){
                    ray_cast_pos = CalcSkipPos(params, ray_cast_pos, ray.d, cur_lod);

                    continue;
                }

                // compute and accumulate color
                // always assert scalar = 0.f has no meanings
                if(scalar > 0.f){
                    float4 mapping_color = ScalarToRGBA(params, scalar, cur_lod);
                    // always assert alpha = 0.f has no contribution
                    if(mapping_color.w > 0.f){
                        mapping_color = CalcShadingColor(params, mapping_color, sampling_pos, cur_lod);
                        // accumulate color radiance
                        ret.color += mapping_color * make_float4(make_float3(mapping_color.w), 1.f) * (1.f - ret.color.w);
                        if(ret.color.w > 0.99f){
                            // view space depth
                            ret.depth = ray_cast_dist;
                            break;
                        }
                    }
                }

                ray_cast_pos += dt * ray.d;
                if(upgrade){
                    dt *= 2.f;
                }
            }

            return ret;
        }
        CUB_GPU uint32_t Float4ToUInt(const float4& f){
            uint32_t ret = (uint32_t)(saturate(f.x) * 255u)
                         | ((uint32_t)(saturate(f.y) * 255u) << 8)
                         | ((uint32_t)(saturate(f.z) * 255u) << 16)
                         | ((uint32_t)(saturate(f.w) * 255u) << 24);
            return ret;
        }
        CUB_KERNEL void CRTVolumeRenderKernel(CTRVolumeRenderKernelParams params){
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            if(x >= params.cu_per_frame_params.frame_width || y >= params.cu_per_frame_params.frame_height)
                return;
            float4 rgba = make_float4((x + 0.5) / params.cu_per_frame_params.frame_width,
                                      (y + 0.5) / params.cu_per_frame_params.frame_height, 0.0, 1.0);
            params.framebuffer.color.at(x, y) = Float4ToUInt(rgba);

            return;

            const unsigned int thread_count = blockDim.x * blockDim.y;
            const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int load_count = (HashTableSize + thread_count - 1) / thread_count;
            const unsigned int thread_end = min(thread_idx + load_count, HashTableSize);
            __shared__ uint4 hash_table[HashTableSize][2];

            // fill shared hash table

            auto& table = params.cu_page_table.table;

            for(int i = thread_idx; i < thread_end; ++i){
                auto& block_uid = table.at(i).first;
                auto& tex_coord = table.at(i).second;
                hash_table[i][0] = {block_uid.x, block_uid.y, block_uid.z, block_uid.w};
                hash_table[i][1] = {tex_coord.sx, tex_coord.sy, tex_coord.sz,
                                    ((uint32_t)tex_coord.tid << 16) | tex_coord.flag};
            }

            __syncthreads();

            Ray ray;
            ray.o = params.cu_per_frame_params.cam_pos;
            float scale = tanf(0.5f * params.cu_per_frame_params.fov);
            float ix = (x + 0.5f) / params.cu_per_frame_params.frame_width;
            float iy = (y + 0.5f) / params.cu_per_frame_params.frame_height;
            ray.d = params.cu_per_frame_params.cam_dir + params.cu_per_frame_params.cam_up * scale * iy
                    + params.cu_per_frame_params.cam_right * scale * ix * params.cu_per_frame_params.frame_w_over_h;

            auto [color, depth] = RayCastVolume(params, hash_table, ray);

            // exposure gamma color-grading tone-mapping...

//            params.framebuffer.color.at(x, y) = color;
//            params.framebuffer.depth.at(x, y) = depth;

        }
    }

    class CRTVolumeRendererPrivate{
    public:

        cub::cu_stream render_stream;

        CTRVolumeRenderKernelParams kernel_params;

        struct{
            Handle<CUDAHostBuffer> tf1d;
            Handle<CUDAHostBuffer> tf2d;
            Handle<CUDATexture> cu_tf_tex;
            Handle<CUDATexture> cu_2d_tf_tex;
            int tf_dim = 0;
        };

        cub::cu_context ctx;

        UnifiedRescUID uid;

        std::mutex g_mtx;

        static constexpr uint32_t shared_mem_size = 32u << 10;

        static UnifiedRescUID GenRescUID(){
            static std::atomic<size_t> g_uid = 1;
            auto uid = g_uid.fetch_add(1);
            return GenUnifiedRescUID(uid, UnifiedRescType::CRTVolRenderer);
        }

        void GenTFTex(int dim){
            if(tf_dim == dim) return;
            tf_dim = dim;
            tf1d.Destroy();
            tf2d.Destroy();
            cu_tf_tex.Destroy();
            cu_2d_tf_tex.Destroy();

            size_t bytes_1d = sizeof(typename TransferFunc::Value) * dim;
            tf1d = NewHandle<CUDAHostBuffer>(RescAccess::Unique, bytes_1d, cub::e_cu_host, ctx);

            size_t bytes_2d = sizeof(typename TransferFunc::Value) * dim * dim;
            tf2d = NewHandle<CUDAHostBuffer>(RescAccess::Unique, bytes_2d, cub::e_cu_host, ctx);

            cub::texture_resc_info resc_info{cub::e_float, 4, {256, 1, 1}};
            cu_tf_tex = NewHandle<CUDATexture>(RescAccess::Unique, resc_info, cub::texture_view_info{}, ctx);

            resc_info.extent = {256, 256, 1};
            cu_2d_tf_tex = NewHandle<CUDATexture>(RescAccess::Unique, resc_info, cub::texture_view_info{}, ctx);
        }
    };

    CRTVolumeRenderer::CRTVolumeRenderer(const CRTVolumeRenderer::CRTVolumeRendererCreateInfo &info) {
        _  = std::make_unique<CRTVolumeRendererPrivate>();
        _->render_stream = cub::cu_stream::null(info.gpu_mem_mgr->_get_cuda_context());

        _->ctx = info.gpu_mem_mgr->_get_cuda_context();

        _->uid = _->GenRescUID();

    }

    CRTVolumeRenderer::~CRTVolumeRenderer(){

    }

    void CRTVolumeRenderer::SetVolume(const VolumeParams& volume_params) {
        _->kernel_params.cu_volume.bound = {{volume_params.bound.low.x,
                                             volume_params.bound.low.y,
                                             volume_params.bound.low.z,},
                                            {volume_params.bound.high.x,
                                             volume_params.bound.high.y,
                                             volume_params.bound.high.z}};
        _->kernel_params.cu_volume.block_length = volume_params.block_length;
        _->kernel_params.cu_volume.padding = volume_params.padding;
        _->kernel_params.cu_volume.voxel_dim = {(float)volume_params.voxel_dim.x,
                                                (float)volume_params.voxel_dim.y,
                                                (float)volume_params.voxel_dim.z};
    }

    void CRTVolumeRenderer::SetRenderParams(const RenderParams& render_params) {
        if(render_params.light.updated){

        }
        if(render_params.lod.updated){
            std::memcpy(_->kernel_params.cu_render_params.lod_policy, render_params.lod.leve_of_dist.LOD, sizeof(float) * MaxLodLevels);
        }
        if(render_params.tf.updated){
            _->GenTFTex(render_params.tf.dim);
            render_params.tf.tf_pts.Gen1DTF(_->tf1d, render_params.tf.dim);
            render_params.tf.tf_pts.Gen2DTF(_->tf1d, _->tf2d, render_params.tf.dim);
            // transfer to texture
            cub::memory_transfer_info info;
            info.width_bytes = sizeof(TransferFunc::Value) * render_params.tf.dim;
            info.height = info.depth = 1;
            cub::cu_memory_transfer(*_->tf1d, *_->cu_tf_tex, info).launch(_->render_stream);
            info.height = render_params.tf.dim;
            cub::cu_memory_transfer(*_->tf2d, *_->cu_2d_tf_tex, info).launch(_->render_stream);
        }
    }


    void CRTVolumeRenderer::SetPerFrameParams(const PerFrameParams& per_frame_params) {
        _->kernel_params.cu_per_frame_params.cam_pos = { Transform(per_frame_params.cam_pos) };
        _->kernel_params.cu_per_frame_params.fov = per_frame_params.fov;
        _->kernel_params.cu_per_frame_params.cam_dir = { Transform(per_frame_params.cam_dir) };
        _->kernel_params.cu_per_frame_params.frame_width = per_frame_params.frame_width;
        _->kernel_params.cu_per_frame_params.cam_right = { Transform(per_frame_params.cam_right) };
        _->kernel_params.cu_per_frame_params.frame_height = per_frame_params.frame_height;
        _->kernel_params.cu_per_frame_params.cam_up = { Transform(per_frame_params.cam_up) };
        _->kernel_params.cu_per_frame_params.frame_w_over_h = per_frame_params.frame_w_over_h;
    }

    void CRTVolumeRenderer::Render(Handle<FrameBuffer> frame) {
        const dim3 tile = {16u, 16u, 1u};
        _->kernel_params.framebuffer.color = frame->color;
        _->kernel_params.framebuffer.depth = frame->depth;
        cub::cu_kernel_launch_info launch_info;
        launch_info.shared_mem_bytes = CRTVolumeRendererPrivate::shared_mem_size;
        launch_info.block_dim = tile;
        launch_info.grid_dim = {(frame->frame_width + tile.x - 1) / tile.x,
                                (frame->frame_height + tile.y - 1) / tile.y, 1};
        void* params[] = {&_->kernel_params};
        auto render_task = cub::cu_kernel::pending(launch_info, &CRTVolumeRenderKernel, params);
        try{
            render_task.launch(_->render_stream).check_error_on_throw();
        }
        catch (const std::exception& err) {
            LOG_ERROR("CRTVolumeRenderer render frame failed : {}", err.what());
        }
    }

    void CRTVolumeRenderer::BindVTexture(VTextureHandle handle, TextureUnit unit) {
        assert(unit >= 0 && unit < MaxCUDATextureCountPerGPU);
        _->kernel_params.cu_vtex[unit] = handle->_get_tex_handle();
    }

    void CRTVolumeRenderer::BindPTBuffer(PTBufferHandle handle) {
        _->kernel_params.cu_page_table.table = handle->view_1d<HashTableItem>(HashTableSize * sizeof(HashTableItem));
    }

    void CRTVolumeRenderer::Lock() {
        _->g_mtx.lock();
    }

    void CRTVolumeRenderer::UnLock() {
        _->g_mtx.unlock();
    }

    UnifiedRescUID CRTVolumeRenderer::GetUID() const {
        return _->uid;
    }


VISER_END