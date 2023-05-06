#undef UTIL_ENABLE_OPENGL

#include <Algorithm/Voxelization.hpp>
#include "../Common/helper_math.h"
#include "../Renderer/Common.hpp"
VISER_BEGIN

    namespace{
        static constexpr int HashTableSize = 1024;
        static constexpr int ThreadsPerBlocks = 64;
        static constexpr int MaxLodLevels = LevelOfDist::MaxLevelCount;
        static constexpr int BlockUIDLodBits = GridVolume::BlockUID::LodBits;
        using HashTableItem = GPUPageTableMgr::PageTableItem;

        CUB_GPU uint32_t GetHashValue(const uint4& key){
            auto p = reinterpret_cast<const uint32_t*>(&key);
            uint32_t v = p[0];
            for(int i = 1; i < 4; i++){
                v = v ^ (p[i] + 0x9e3779b9u + (v << 6) + (v >> 2));
            }
            return v;
        }

#define INVALID_VALUE uint4{0, 0, 0, 0}
        CUB_GPU void Update(const uint4& key, const uint4& val, uint4 hash_table[][2]){
            uint32_t hash_v = GetHashValue(key);

            auto pos = hash_v % HashTableSize;
//            printf("Update hash %u pos %d key %d %d %d %d val %d %d %d %d\n",
//                   hash_v, pos,
//                   key.x, key.y, key.z, key.w,
//                   val.x, val.y, val.z, val.w);
            int i = 0;
            bool positive = false;
            while(i < HashTableSize){
                int ii = i * i;
                pos += positive ? ii : -ii;
                pos %= HashTableSize;
                if(hash_table[pos][0] == key){

                    hash_table[pos][1] = val;
                    return;
                }
                if(positive)
                    ++i;
                positive = !positive;

            }
        }
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

        struct CUDAVolumeParams{
            cuda::AABB bound;
            float3 space;
            uint3 voxel_dim;
            uint32_t block_length;
            uint32_t padding;
        };

        struct CUDAPageTable{
            CUDABufferView1D<HashTableItem> table;
        };
        struct CUDASWCVoxelizeParams{
            uint32_t lod;
            uint32_t segment_count;
        };
        struct Lock{
            int mutex;
            CUB_CPU_GPU Lock(){};
            CUB_GPU void init(){
                mutex = 0;
            }
            CUB_GPU void lock(){
                while(atomicCAS(&mutex, 0, 1) != 0);
            }
            CUB_GPU void unlock(){
                atomicExch(&mutex,0);
            }
        };
        struct SWCVoxelizeKernelParams{
            cudaSurfaceObject_t cu_vsurf[MaxCUDATextureCountPerGPU];
            CUDABufferView3D<uint8_t> cu_vbuffer[MaxCUDATextureCountPerGPU];
            CUDAPageTable cu_page_table;
            CUDAVolumeParams cu_vol_params;
            CUDASWCVoxelizeParams cu_swc_v_params;
            CUDABufferView1D<std::pair<Float4,Float4>> cu_segments;
            CUDABufferView1D<Lock> g_lk;
        };
        CUB_GPU float DecodeVoxelValue(const SWCVoxelizeKernelParams& params, uint8_t voxel){
            float dist_threshold = length(params.cu_vol_params.space);
            return dist_threshold * (1.f - voxel / 255.f);
//            if(voxel <= 127) return dist_threshold - voxel * dist_threshold / 127.f;
//            return -(voxel - 127.f) * dist_threshold / 128.f;
        }

        __device__ static inline uint8_t atomicCAS(uint8_t* address, uint8_t expected, uint8_t desired) {
            size_t long_address_modulo = (size_t) address & 3;
            auto* base_address = (unsigned int*) ((uint8_t*) address - long_address_modulo);
            unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};

            unsigned int sel = selectors[long_address_modulo];
            unsigned int long_old, long_assumed, long_val, replacement;
            uint8_t old;

            long_val = (unsigned int) desired;
            long_old = *base_address;
            do {
                long_assumed = long_old;
                replacement = __byte_perm(long_old, long_val, sel);
                long_old = ::atomicCAS(base_address, long_assumed, replacement);
                old = (uint8_t) ((long_old >> (long_address_modulo * 8)) & 0x000000ff);
            } while (expected == old && long_assumed != long_old);

            return old;
        }

        CUB_GPU void VirtualStore(SWCVoxelizeKernelParams& params,
                                  uint4 hash_table[][2], Lock& lk,
                                  uint8_t val,
                                  uint3 voxel_coord,
                                  uint32_t lod){
            uint32_t block_length = params.cu_vol_params.block_length;
            uint32_t lod_block_length = block_length << lod;
            uint3 block_uid = voxel_coord / lod_block_length;
            auto padding = params.cu_vol_params.padding;
            uint3 offset_in_block = (voxel_coord - block_uid * lod_block_length) / (1 << lod) + padding;

            auto voxel_store = [&](uint3 block_uid, uint3 offset_in_block){
                uint4 key = make_uint4(block_uid, lod | (VolumeBlock_IsSWC << BlockUIDLodBits));
                uint4 tex_coord = Query(key, hash_table);
                uint32_t tid = (tex_coord.w >> 16) & 0xffff;
                uint3 coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
                if((tex_coord.w & 0xffff & TexCoordFlag_IsValid)
                    //下面的判断条件其实可以不用也没事 如果前面获取真正需要mc的数据块正确的话
                    && (tex_coord.w & 0xffff & TexCoordFlag_IsSWC)){
                    uint32_t block_size = params.cu_vol_params.block_length + padding * 2;
                    auto pos = coord * block_size + offset_in_block;// !! + params.cu_vol_params.padding;
                    //                printf("block %d %d %d %d tex_coord %d %d %d, %d pos %d %d %d",key.x, key.y, key.z, key.w,
                    //                       tex_coord.x, tex_coord.y, tex_coord.z, tid,
                    //                       pos.x, pos.y, pos.z);
                    //                return;
#ifdef USE_SDF

                    auto od_ptr = &params.cu_vbuffer[tid].at(pos.x, pos.y, pos.z);
                    auto od_val = *od_ptr;
//                    float od_sd = DecodeVoxelValue(params, od_val);
                    auto update = [](uint8_t od, uint8_t nw){
                        return nw > od;
//                        if(od >= 127 && nw <= 127 || od >= 127 && nw >= od) return false;
//                        if(od <= 127 && nw <= od) return false;
//                        return true;
                    };
                    while(true){
                        if(!update(od_val, val)) break;
                        if(atomicCAS(od_ptr, od_val, val) == od_val) break;
                        od_val = *od_ptr;
                    }

#else
                    surf3Dwrite(val, params.cu_vsurf[tid], pos.x, pos.y, pos.z);
#endif
                    //not necessary now
                    if((tex_coord.w & 0xffff & TexCoordFlag_IsSWCV) == 0){
                        tex_coord.w |= TexCoordFlag_IsSWCV;
                        //lock
                        lk.lock();
                        Update(key, tex_coord, hash_table);
                        lk.unlock();
                    }
                }
            };

            auto lod_block_dim = (params.cu_vol_params.voxel_dim + lod_block_length - 1) / lod_block_length;

            auto block_uid_valid = [&](uint3 block_uid){
                return block_uid.x < lod_block_dim.x
                    && block_uid.y < lod_block_dim.y
                    && block_uid.z < lod_block_dim.z;
            };

//            if(block_uid_valid(block_uid)) voxel_store(block_uid, offset_in_block);
            const auto _offset_in_block = offset_in_block;
            const auto _block_uid = block_uid;

            auto xyz_store = [_block_uid, _offset_in_block, padding, block_length,
            &block_uid_valid, &voxel_store](int x, int y, int z) {
                auto block_uid = _block_uid;
                auto offset_in_block = _offset_in_block;
                int x_ok = 1, y_ok = 1, z_ok = 1;
                if(x){
                    if(offset_in_block.x >= block_length){
                        ++block_uid.x;
                        offset_in_block.x -= block_length;
                    }
                    else if(offset_in_block.x < 2u * padding && block_uid.x > 0){
                        --block_uid.x;
                        offset_in_block.x += block_length;
                    }
                    else{
                        x_ok = 0;
                    }
                }
                if(y){
                    if(offset_in_block.y >= block_length){
                        ++block_uid.y;
                        offset_in_block.y -= block_length;
                    }
                    else if(offset_in_block.y < 2u * padding && block_uid.y > 0){
                        --block_uid.y;
                        offset_in_block.y += block_length;
                    }
                    else{
                        y_ok = 0;
                    }
                }
                if(z){
                    if(offset_in_block.z >= block_length){
                        ++block_uid.z;
                        offset_in_block.z -= block_length;
                    }
                    else if(offset_in_block.z < 2u * padding && block_uid.z > 0){
                        --block_uid.z;
                        offset_in_block.z += block_length;
                    }
                    else{
                        z_ok = 0;
                    }
                }
                auto ok = [](int a, int b)->bool{
                    if(a) return b;
                    return true;
                };
                if(!ok(x, x_ok) || !ok(y, y_ok) || !ok(z, z_ok)) return;
                if(block_uid_valid(block_uid)){
//                    if(x + y + z > 0)
//                        printf("origin uid : %d %d %d, "
//                               "origin offset: %d %d %d, "
//                               "xyz: %d %d %d, "
//                               "uid : %d %d %d, "
//                               "offset : %d %d %d\n",
//                               _block_uid.x, _block_uid.y, _block_uid.z,
//                               _offset_in_block.x, _offset_in_block.y, _offset_in_block.z,
//                               x, y, z,
//                               block_uid.x, block_uid.y, block_uid.z,
//                               offset_in_block.x, offset_in_block.y, offset_in_block.z);
                    voxel_store(block_uid, offset_in_block);
                }
            };

            xyz_store(0, 0, 0);
            xyz_store(1, 0, 0);
            xyz_store(0, 1, 0);
            xyz_store(0, 0, 1);
            xyz_store(0, 1, 1);
            xyz_store(1, 0, 1);
            xyz_store(1, 1 ,0);
            xyz_store(1, 1, 1);
        }
        CUB_GPU cuda::AABB_UI ComputeSegmentAABB(const SWCVoxelizeKernelParams& params,
                                              float3 pt_a_pos, float pt_a_r,
                                              float3 pt_b_pos, float pt_b_r
                                              ){
            cuda::AABB_UI ret;
            float3 low = fminf(pt_a_pos, pt_b_pos);
            float3 high = fmaxf(pt_a_pos, pt_b_pos);
            low = fminf(low,  pt_a_pos - pt_a_r);
            low = fminf(low,  pt_b_pos - pt_b_r);
            high = fmaxf(high + pt_a_r, high + pt_b_r);
            high = fmaxf(high, pt_a_pos + pt_a_r);
            high = fmaxf(high, pt_b_pos + pt_b_r);
            low = (low - params.cu_vol_params.bound.low) / (params.cu_vol_params.bound.high - params.cu_vol_params.bound.low);
            high = (high - params.cu_vol_params.bound.low) / (params.cu_vol_params.bound.high - params.cu_vol_params.bound.low);
            low *= params.cu_vol_params.voxel_dim - make_uint3(1);
            high *= params.cu_vol_params.voxel_dim - make_uint3(1);
            ret.low.x = max(0, int(low.x - 2));
            ret.low.y = max(0, int(low.y - 2));
            ret.low.z = max(0, int(low.z - 2));
            ret.high.x = min(params.cu_vol_params.voxel_dim.x - 1, int(high.x + 3));
            ret.high.y = min(params.cu_vol_params.voxel_dim.y - 1, int(high.y + 3));
            ret.high.z = min(params.cu_vol_params.voxel_dim.z - 1, int(high.z + 3));
            return ret;
        }

        //一个block负责一条线，即两个点之间的体素化，因此需要控制输入的神经元每一段的长度适中，使得其包围盒的大小在一定范围
        CUB_KERNEL void SWCVoxelizeKernel(SWCVoxelizeKernelParams params){
            __shared__ uint4 hash_table[HashTableSize][2];
            __shared__ Lock lk;

            const unsigned int block_idx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
            if(block_idx >= params.cu_swc_v_params.segment_count) return;

            const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
            const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
            const unsigned int load_count = (HashTableSize + thread_count - 1) / thread_count;
            const unsigned int thread_beg = thread_idx * load_count;
            const unsigned int thread_end = min(thread_beg + load_count, HashTableSize);

            if(thread_idx == 0) lk.init();

            if(block_idx == 0 && thread_idx == 0) params.g_lk.at(0).init();


            __syncthreads();

            // fill shared hash table

            auto& table = params.cu_page_table.table;
            for(int i = 0; i < HashTableSize; ++i){
                auto& block_uid = table.at(i).first;
                auto& tex_coord = table.at(i).second;
                hash_table[i][0] = uint4{block_uid.x, block_uid.y, block_uid.z, block_uid.w};
                hash_table[i][1] = uint4{tex_coord.sx, tex_coord.sy, tex_coord.sz,
                                         ((uint32_t)tex_coord.tid << 16) | tex_coord.flag};
            }

            __syncthreads();


            // 一个block里的所有thread都是为特定一条的segment体素化服务的

            auto& pt = params.cu_segments.at(block_idx);
            float3 pt_a_pos = make_float3(pt.first.x, pt.first.y, pt.first.z);
            float pt_a_r = pt.first.w;
            float3 pt_b_pos = make_float3(pt.second.x, pt.second.y, pt.second.z);
            float pt_b_r = pt.second.w;
            float3 a_to_b = pt_b_pos - pt_a_pos;
            float3 ab = normalize(a_to_b);
            float ab_dist = dot(ab, a_to_b);

            auto inside_segment = [&](uint3 voxel_coord)->bool{
                float3 pt_c_pos = (voxel_coord + make_float3(0.5f)) * params.cu_vol_params.space;

                float dist_a = length(pt_c_pos - pt_a_pos);
                float dist_b = length(pt_c_pos - pt_b_pos);
                if(dist_a <= pt_a_r) return true;
                if(dist_b <= pt_b_r) return true;
//                return false;
                float proj_a_to_c = dot(pt_c_pos - pt_a_pos, ab);
                bool between = proj_a_to_c > 0 && proj_a_to_c < ab_dist;
                if(!between) return false;
                float u = proj_a_to_c / ab_dist;
                float R = (1.f - u) * pt_a_r + u * pt_b_r;
                // 点到直线的距离公式 利用叉乘得到
                float c_to_ab_dist = length(cross(pt_c_pos - pt_a_pos, ab));
                return c_to_ab_dist <= R;
            };

            float dist_threshold = length(params.cu_vol_params.space);

            float l = sqrt((pt_a_r - pt_b_r) * (pt_a_r - pt_b_r) + dot(a_to_b, a_to_b));

            auto inside_segment_neighborhood = [&, pt_a_r = pt_a_r - 0.2f * dist_threshold,
                                                   pt_b_r = pt_b_r - 0.2f * dist_threshold]
                                                (uint3 voxel_coord)->float{
                float3 pt_c_pos = (voxel_coord + make_float3(0.5f)) * params.cu_vol_params.space;
                float3 N = normalize(cross(pt_b_pos - pt_c_pos, pt_a_pos - pt_c_pos));
                auto calc_p = [&](const float3& P, const float3& O, const float3& L, float R)->float3{
                    float3 PO = O - P;
                    float3 PA = dot(PO, L) * L;
                    float3 OA = PA - PO;
                    float len_OA = length(OA);
                    if(len_OA < FLT_EPSILON){
                        OA = normalize(cross(N, L));
                        len_OA = 1.f;
                    }
                    float3 OC = R / len_OA * OA;
                    return O + OC;
                };

                float3 C = calc_p(pt_c_pos, pt_a_pos, make_float3(0.f) - ab, pt_a_r);
                float3 D = calc_p(pt_c_pos, pt_b_pos,  ab, pt_b_r);

                float3 DC = normalize(C - D);

                float3 X = cross(DC, N);
                float3 DP = pt_c_pos - D;
                float sd = dot(DP, X);
                float h = dot(DP, DC);
                if(h > 0 && h < l){
                    return -sd;
                }
                else if(h <= 0){
//                    return -pt_b_r;
                    return length(pt_c_pos - pt_b_pos) - pt_b_r;
                }
                else{
//                    return -pt_a_r;
                    return length(pt_c_pos - pt_a_pos) - pt_a_r;
                }
            };

            auto signed_dist_to_uint8 = [dist_threshold, inv_dist_threshold = 1.f / dist_threshold](float dist)->uint8_t{
                if(dist <= 0.f) return 255;
                if(dist >= dist_threshold) return 0;
                return (dist_threshold - dist) * inv_dist_threshold * 255;
//                if(dist >= 0) return max(int((dist_threshold - dist) * inv_dist_threshold * 127), 0);
//                if(dist < 0) return min(255, 127 + int(128 * (-dist) * inv_dist_threshold));
            };

            // 计算这条线占据的voxel范围，返回的必定在有效范围内
//            printf("pt a %f %f %f %f, pt b %f %f %f %f\n",
//                   pt_a_pos.x, pt_a_pos.y, pt_a_pos.z, pt_a_r,
//                   pt_b_pos.x, pt_b_pos.y, pt_b_pos.z, pt_b_r);
            auto [low, high] = ComputeSegmentAABB(params, pt_a_pos, pt_a_r, pt_b_pos, pt_b_r);
//            if(pt_a_r == 0.01f)
//                printf("low %d %d %d, high %d %d %d\n", low.x, low.y, low.z, high.x, high.y, high.z);
            uint3 box_dim = high - low + 1;
            uint32_t box_voxel_count = box_dim.x * box_dim.y * box_dim.z;
            uint32_t thread_voxel_num = (box_voxel_count + thread_count - 1) / thread_count;
//            if(pt_a_r == 0.01f)
//                printf("threadIdx %d, thread_voxel_num %d, box_voxel_count %d\n",
//                   thread_idx, thread_voxel_num, box_voxel_count);


            for(uint32_t i = 0; i < thread_voxel_num; i++){
                uint32_t box_voxel_idx = thread_idx * thread_voxel_num + i;
                if(box_voxel_idx >= box_voxel_count) break;
                // 计算得到当前处理的体素坐标
                uint32_t box_voxel_z = box_voxel_idx / (box_dim.x * box_dim.y);
                uint32_t box_voxel_y = (box_voxel_idx - box_voxel_z * box_dim.x * box_dim.y) / box_dim.x;
                uint32_t box_voxel_x = box_voxel_idx - box_voxel_z * box_dim.x * box_dim.y - box_voxel_y * box_dim.x;
                uint3 voxel_coord = low + make_uint3(box_voxel_x, box_voxel_y, box_voxel_z);
#ifdef USE_SDF
                VirtualStore(params, hash_table, lk,
                             signed_dist_to_uint8(inside_segment_neighborhood(voxel_coord)),
                             voxel_coord, params.cu_swc_v_params.lod);
#else
                if(inside_segment(voxel_coord)){
                    VirtualStore(params, hash_table, lk, SWCVoxelVal, voxel_coord, params.cu_swc_v_params.lod);
                }
#endif

            }
//            if(thread_idx == 0){
//                printf("low: %d %d %d, high: %d %d %d, dim : %d %d %d,"
//                       "p_a: %f %f %f %f, p_b: %f %f %f %f\n", low.x, low.y, low.z, high.x, high.y, high.z,
//                       box_dim.x ,box_dim.y, box_dim.z,
//                       pt_a_pos.x, pt_a_pos.y, pt_a_pos.z, pt_a_r,
//                       pt_b_pos.x, pt_b_pos.y, pt_b_pos.z, pt_b_r);
//
//            }
            __syncthreads();
            // write back from shared to global memory page table
            if(thread_idx == 0){
                auto& g_lk = params.g_lk.at(0);
                g_lk.lock();

                for(int i = 0; i < HashTableSize; ++i){
                    auto& block_uid = table.at(i).first;
                    auto& tex_coord = table.at(i).second;
                    block_uid = GridVolume::BlockUID(hash_table[i][0].x,
                                                     hash_table[i][0].y,
                                                     hash_table[i][0].z,
                                                     hash_table[i][0].w);
                    tex_coord.sx = hash_table[i][1].x;
                    tex_coord.sy = hash_table[i][1].y;
                    tex_coord.sz = hash_table[i][1].z;
                    tex_coord.tid = hash_table[i][1].w >> 16;
                    tex_coord.flag |= hash_table[i][1].w & 0xffff;
                }

                g_lk.unlock();
            }
        }
    }


    class SWCVoxelizerPrivate{
    public:
        CUDAContext ctx;

        CUDAStream compute_stream;

        SWCVoxelizeKernelParams kernel_params;

        Handle<CUDABuffer> cu_ptrs;
        CUDABufferView1D<SWCSegment> cu_ptrs_view;

        Handle<CUDABuffer> cu_glk;

        Handle<CUDABuffer> cu_clear_buffer;

        std::mutex mtx;

        UnifiedRescUID uid;

        static UnifiedRescUID GenRescUID(){
            static std::atomic<size_t> g_uid = 1;
            auto uid = g_uid.fetch_add(1);
            return GenUnifiedRescUID(uid, UnifiedRescType::SWCVoxelizeAlgo);
        }
    };

    SWCVoxelizer::SWCVoxelizer(const VoxelizerCreateInfo& info) {
        _ = std::make_unique<SWCVoxelizerPrivate>();

        _->uid = _->GenRescUID();

        _->ctx = info.gpu_mem_mgr._get_ptr()->_get_cuda_context();

        //暂时使用null流
        _->compute_stream = cub::cu_stream::null(info.gpu_mem_mgr._get_ptr()->_get_cuda_context());

        _->cu_ptrs = NewHandle<CUDABuffer>(ResourceType::Buffer, info.max_segment_count * sizeof(SWCSegment), cub::e_cu_device, _->ctx);
        _->cu_ptrs_view = _->cu_ptrs->view_1d<SWCSegment>(info.max_segment_count);
        _->kernel_params.cu_segments = _->cu_ptrs_view;

        _->cu_glk = NewHandle<CUDABuffer>(ResourceType::Buffer, sizeof(viser::Lock), cub::e_cu_device, _->ctx);
        _->kernel_params.g_lk = _->cu_glk->view_1d<viser::Lock>(1);
    }

    SWCVoxelizer::~SWCVoxelizer() {

    }

    void SWCVoxelizer::Lock() {
        _->mtx.lock();
    }

    void SWCVoxelizer::UnLock() {
        _->mtx.unlock();
    }

    UnifiedRescUID SWCVoxelizer::GetUID() const {
        return _->uid;
    }

    void SWCVoxelizer::Run(const SWCVoxelizeAlgoParams &params) {
        size_t segment_count = params.ptrs.size();
        dim3 grid(segment_count, 1, 1);
        dim3 threads(8, 8, 8);
        if(grid.x > 65535){
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
        if(grid.y > 65535){
            throw std::runtime_error("Too many segments for SWCVoxelizer to Run");
        }

        _->kernel_params.cu_swc_v_params.lod = params.lod;
        _->kernel_params.cu_swc_v_params.segment_count = segment_count;


        try{
            cub::memory_transfer_info m_info;
            m_info.width_bytes = segment_count * sizeof(SWCSegment);

            cub::cu_memory_transfer(params.ptrs, _->cu_ptrs_view, m_info)
            .launch(_->compute_stream).check_error_on_throw();

            cub::cu_kernel_launch_info info{grid, threads};
            void* launch_params[] = {&_->kernel_params};
            cub::cu_kernel::pending(info, &SWCVoxelizeKernel, launch_params)
            .launch(_->compute_stream).check_error_on_throw();
        }
        catch (const std::exception& err) {
            LOG_ERROR("SWCVoxolizer run failed : {}", err.what());
        }
    }

    void SWCVoxelizer::BindVTexture(VTextureHandle handle, TextureUnit unit) {
        assert(unit >= 0 && unit < MaxCUDATextureCountPerGPU);
//        if(_->kernel_params.cu_vsurf[unit] != 0){
//            cudaDestroySurfaceObject(_->kernel_params.cu_vsurf[unit]);
//        }
//        else{
            _->kernel_params.cu_vsurf[unit] = handle->as_surface();
//        }
    }

    void SWCVoxelizer::BindPTBuffer(PTBufferHandle handle) {
        _->kernel_params.cu_page_table.table = handle->view_1d<HashTableItem>(HashTableSize * sizeof(HashTableItem));
    }

    void SWCVoxelizer::SetVolume(const VolumeParams &volume_params) {
        _->kernel_params.cu_vol_params.space = {
                volume_params.space.x,
                volume_params.space.y,
                volume_params.space.z
        };
        _->kernel_params.cu_vol_params.block_length = volume_params.block_length;
        _->kernel_params.cu_vol_params.padding = volume_params.padding;
        _->kernel_params.cu_vol_params.voxel_dim = {
                volume_params.voxel_dim.x,
                volume_params.voxel_dim.y,
                volume_params.voxel_dim.z
        };
        _->kernel_params.cu_vol_params.bound = {{volume_params.bound.low.x,
                                                    volume_params.bound.low.y,
                                                    volume_params.bound.low.z,},
                                            {volume_params.bound.high.x,
                                                    volume_params.bound.high.y,
                                                    volume_params.bound.high.z}};
    }

#ifdef USE_LINEAR_BUFFER_FOR_TEXTURE
    void SWCVoxelizer::BindVBuffer(CUDABufferView3D<uint8_t> view, TextureUnit unit)
    {
        assert(unit >= 0 && unit < MaxCUDATextureCountPerGPU);
        _->kernel_params.cu_vbuffer[unit] = view;
    }
#endif

    VISER_END


