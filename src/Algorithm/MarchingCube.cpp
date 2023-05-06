#undef UTIL_ENABLE_OPENGL

#include <Algorithm/MarchingCube.hpp>
#include "LookUpTable.cuh"
#include "../Common/helper_math.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define SKIP_EMPTY_VOXELS

#define USE_SHARED_CUBE_VOXELS

VISER_BEGIN

    namespace{

        static constexpr int HashTableSize = 1024;
        static constexpr int ThreadsPerBlocks = 64;
        static constexpr int MaxLodLevels = LevelOfDist::MaxLevelCount;
        using HashTableItem = GPUPageTableMgr::PageTableItem;

        __constant__ unsigned char TriNumTable[43] = {
                0, // 0
                1, // 1
                2, // 2
                2, 4, // 3.1  3.2
                2, 6, // 4.1  4.2
                3, // 5
                3, 9, 5, // 6.1  6.2  6.3
                3, 5, 5, 9, 5, 9, 9, 5, 9, // 7.1  7.2  7.3  7.4  7.5  7.6  7.7  7.8  7.9
                2, // 8
                4, // 9
                4, 8, 8, 8, 4, // 10.1  10.2  10.3  10.4  10.5
                4, // 11
                4, 8, 8, 8, 4, // 12.1  12.2  12.3  12.4  12.5
                4, 6, 10, 12, 6, 10, 10, 6, 4, // 13.1  13.2  13.3  13.4  13.5  13.6  13.7  13.8  13.9
                4 // 14
        };

        __constant__ unsigned char MyCaseTable[43] = {
                0,
                16,
                24,
                24,
                24,
                8,
                8,
                48,
                48,
                48,
                48,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                6,
                8,
                6,
                6,
                6,
                6,
                6,
                12,
                24,
                24,
                24,
                24,
                24,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                12
        };

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

        struct CUDAPageTable{
            CUDABufferView1D<HashTableItem> table;
        };
        struct CUDAVolumeParams{
            float3 space;
            uint3 voxel_dim;
            uint32_t block_length;
            uint32_t padding;
        };
        struct CUDAMCAlgoParams{
            uint3 origin; uint32_t lod;
            uint3 shape;
            float isovalue;
        };
        struct MCKernelParams{
            CUDABufferView3D<uint8_t> cu_vbuf[MaxCUDATextureCountPerGPU];
            cudaTextureObject_t cu_vtex[MaxCUDATextureCountPerGPU];

            CUDAPageTable cu_page_table;
            CUDAVolumeParams cu_vol_params;
            CUDAMCAlgoParams cu_mc_params;

            CUDABufferView3D<uint32_t> vol_code;
            CUDABufferView3D<uint32_t> vertex_num;
            CUDABufferView3D<uint32_t> cube_occupied;
            // 43 种情况对应的edge table起始地址
            CUDABufferView1D<float3> vertex_pos;
            uint32_t gen_vert_num;

            uint32_t max_vert_num;

            //存储每一个体素之前总共产生的顶点数量
            CUDABufferView1D<uint32_t> num_verts_scanned;

            uint32_t occupied_voxel_num;//实际生成三角形的体素数量

            CUDABufferView1D<uint32_t> cube_occupied_scanned;
            CUDABufferView1D<uint32_t> compacted_voxel_array;
        };

        CUB_GPU float DecodeVoxelValue(float dist_threshold, const uint8_t voxel){
            if(voxel == 255) return 0.f;
            return dist_threshold * (1.f - voxel / 255.f);
//            if(voxel <= 127) return dist_threshold - voxel * dist_threshold / 127.f;
//            return -(voxel - 127.f) * dist_threshold / 128.f;
        }

        CUB_GPU float VirtualSampling(MCKernelParams& params,
                              uint4 hash_table[][2],
                              uint3 voxel_coord, uint32_t lod, uint32_t xyz){
            uint32_t lod_block_length = params.cu_vol_params.block_length << lod;
            uint3 block_uid = voxel_coord / lod_block_length;
            uint3 offset_in_block = (voxel_coord - block_uid * lod_block_length) / (1 << lod);
            uint4 key = make_uint4(block_uid, lod | (VolumeBlock_IsSWC << 8));
            uint4 tex_coord = Query(key, hash_table);
            uint32_t tid = (tex_coord.w >> 16) & 0xffff;
            uint3 coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
#ifdef USE_SDF
            uint8_t voxel_val;
#endif
            float ret = 0.f;

            if((tex_coord.w & 0xffff) & TexCoordFlag_IsValid){
                uint32_t block_size = params.cu_vol_params.block_length + params.cu_vol_params.padding * 2;
                auto pos = coord * block_size + offset_in_block + params.cu_vol_params.padding;
#ifdef USE_SDF
#ifndef USE_LINEAR_BUFFER_FOR_TEXTURE
                voxel_val = tex3D<uint8_t>(params.cu_vtex[tid], pos.x, pos.y, pos.z);
#else
                voxel_val = params.cu_vbuf[tid].at(pos.x, pos.y, pos.z);
#endif
#else
                ret = tex3D<float>(params.cu_vtex[tid], pos.x, pos.y, pos.z);
#endif

            }
            else{
                auto padding = params.cu_vol_params.padding;
                auto block_length = params.cu_vol_params.block_length;
                auto _block_uid = block_uid;
                auto _offset = offset_in_block;

                auto voxel_read = [&](uint3 block_uid, uint3 offset_in_block){
                    key = make_uint4(block_uid, lod | (VolumeBlock_IsSWC << 8));
                    tex_coord = Query(key, hash_table);
                    tid = (tex_coord.w >> 16) & 0xffff;
                    coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
                    if((tex_coord.w & 0xffff) & TexCoordFlag_IsValid){
                        uint32_t block_size = params.cu_vol_params.block_length + params.cu_vol_params.padding * 2;
                        auto pos = coord * block_size + offset_in_block + params.cu_vol_params.padding;
#ifdef USE_SDF
#ifndef USE_LINEAR_BUFFER_FOR_TEXTURE
                        voxel_val = tex3D<uint8_t>(params.cu_vtex[tid], pos.x, pos.y, pos.z);
#else
                        voxel_val = params.cu_vbuf[tid].at(pos.x, pos.y, pos.z);
#endif
#else
                        ret = tex3D<float>(params.cu_vtex[tid], pos.x, pos.y, pos.z);
#endif
                    }
                };

                auto lod_block_dim = (params.cu_vol_params.voxel_dim + lod_block_length - 1) / lod_block_length;

                auto block_uid_valid = [&](uint3 block_uid){
                    return block_uid.x < lod_block_dim.x
                           && block_uid.y < lod_block_dim.y
                           && block_uid.z < lod_block_dim.z;
                };
                uint32_t x = xyz & 0b100u;
                uint32_t y = xyz & 0b010u;
                uint32_t z = xyz & 0b001u;
                if(x && offset_in_block.x < padding && block_uid.x > 0){
                    --block_uid.x;
                    offset_in_block.x += block_length;
                }
                if(y && offset_in_block.y < padding && block_uid.y > 0){
                    --block_uid.y;
                    offset_in_block.y += block_length;
                }
                if(z && offset_in_block.z < padding && block_uid.z > 0){
                    --block_uid.z;
                    offset_in_block.z += block_length;
                }
                if(block_uid_valid(block_uid)) voxel_read(block_uid, offset_in_block);

            }
#ifdef USE_SDF
            ret = DecodeVoxelValue(length(params.cu_vol_params.space), voxel_val);
#endif
            return ret;
        }

        CUB_GPU bool TestFace(const float field[8], int f){
            // 渐近线测试所用的四个参数
            int A, B, C, D;
            // 注意有的面需要将结果取反，这个时候 f 是负数
            // 参考 Figure 6 第 3 个图，这个顺序 ABCD 的应该是作者自己确定的，论文里面没写，这个顺序跟测试是否带正负号相关
            switch (f) {
                case -1: case 1: A = 0, B = 4, C = 5, D = 1; break;
                case -2: case 2: A = 1, B = 5, C = 6, D = 2; break;
                case -3: case 3: A = 2, B = 6, C = 7, D = 3; break;
                case -4: case 4: A = 3, B = 7, C = 4, D = 0; break;
                case -5: case 5: A = 0, B = 3, C = 2, D = 1; break;
                case -6: case 6: A = 4, B = 7, C = 6, D = 5; break;
                default:{

                }
            }
            if(fabs(field[A] * field[C] - field[B] * field[D]) < FLT_EPSILON)
                return f >= 0;
            return f * field[A] * (field[A] * field[C] - field[B] * field[D]) >= 0;
        }

        CUB_GPU bool TestInterior(const float field[8], int caseIdx, int alongEdgeIdx, int edgeIdx){
            float t, At = 0, Bt = 0, Ct = 0, Dt = 0, a, b;
            switch (caseIdx) {
                // 强对称性，直接计算
                case 4:
                case 10:
                    a = (field[4] - field[0]) * (field[6] - field[2]) - (field[7] - field[3]) * (field[5] - field[1]);
                    b = field[2] * (field[4] - field[0]) + field[0] * (field[6] - field[2]) - field[1] * (field[7] - field[3]) - field[3] * (field[5] - field[1]);
                    t = -b / (2 * a);
                    if (t > 0 || t < 1) return alongEdgeIdx > 0;
                    At = field[0] + (field[4] - field[0]) * t;
                    Bt = field[3] + (field[7] - field[3]) * t;
                    Ct = field[2] + (field[6] - field[2]) * t;
                    Dt = field[1] + (field[5] - field[1]) * t;
                    break;
                    // 没有强对称性，根据 edgeIdx 计算
                case 6:
                case 7:
                case 12:
                case 13:
                    switch (edgeIdx) {
                        case 0:
                            t = field[0] / (field[0] - field[1]);
                            At = 0;
                            Bt = field[3] + (field[2] - field[3]) * t;
                            Ct = field[7] + (field[6] - field[7]) * t;
                            Dt = field[4] + (field[5] - field[4]) * t;
                            break;
                        case 1:
                            t = field[1] / (field[1] - field[2]);
                            At = 0;
                            Bt = field[0] + (field[3] - field[0]) * t;
                            Ct = field[4] + (field[7] - field[4]) * t;
                            Dt = field[5] + (field[6] - field[5]) * t;
                            break;
                        case 2:
                            t = field[2] / (field[2] - field[3]);
                            At = 0;
                            Bt = field[1] + (field[0] - field[1]) * t;
                            Ct = field[5] + (field[4] - field[5]) * t;
                            Dt = field[6] + (field[7] - field[6]) * t;
                            break;
                        case 3:
                            t = field[3] / (field[3] - field[0]);
                            At = 0;
                            Bt = field[2] + (field[1] - field[2]) * t;
                            Ct = field[6] + (field[5] - field[6]) * t;
                            Dt = field[7] + (field[4] - field[7]) * t;
                            break;
                        case 4:
                            t = field[4] / (field[4] - field[5]);
                            At = 0;
                            Bt = field[7] + (field[6] - field[7]) * t;
                            Ct = field[3] + (field[2] - field[3]) * t;
                            Dt = field[0] + (field[1] - field[0]) * t;
                            break;
                        case 5:
                            t = field[5] / (field[5] - field[6]);
                            At = 0;
                            Bt = field[4] + (field[7] - field[4]) * t;
                            Ct = field[0] + (field[3] - field[0]) * t;
                            Dt = field[1] + (field[2] - field[1]) * t;
                            break;
                        case 6:
                            t = field[6] / (field[6] - field[7]);
                            At = 0;
                            Bt = field[5] + (field[4] - field[5]) * t;
                            Ct = field[1] + (field[0] - field[1]) * t;
                            Dt = field[2] + (field[3] - field[2]) * t;
                            break;
                        case 7:
                            t = field[7] / (field[7] - field[4]);
                            At = 0;
                            Bt = field[6] + (field[5] - field[6]) * t;
                            Ct = field[2] + (field[1] - field[2]) * t;
                            Dt = field[3] + (field[0] - field[3]) * t;
                            break;
                        case 8:
                            t = field[0] / (field[0] - field[4]);
                            At = 0;
                            Bt = field[3] + (field[7] - field[3]) * t;
                            Ct = field[2] + (field[6] - field[2]) * t;
                            Dt = field[1] + (field[5] - field[1]) * t;
                            break;
                        case 9:
                            t = field[1] / (field[1] - field[5]);
                            At = 0;
                            Bt = field[0] + (field[4] - field[0]) * t;
                            Ct = field[3] + (field[7] - field[3]) * t;
                            Dt = field[2] + (field[6] - field[2]) * t;
                            break;
                        case 10:
                            t = field[2] / (field[2] - field[6]);
                            At = 0;
                            Bt = field[1] + (field[5] - field[1]) * t;
                            Ct = field[0] + (field[4] - field[0]) * t;
                            Dt = field[3] + (field[7] - field[3]) * t;
                            break;
                        case 11:
                            t = field[3] / (field[3] - field[7]);
                            At = 0;
                            Bt = field[2] + (field[6] - field[2]) * t;
                            Ct = field[1] + (field[5] - field[1]) * t;
                            Dt = field[0] + (field[4] - field[0]) * t;
                            break;
                        default:
//                            std::cout << "testInterior got wrong edgeIdx: " << edgeIdx << std::endl;
//                            assert(false);
                            break;
                    }
                    break;
                default:{
//                    std::cout << "testInterior got wrong caseIdx: " << caseIdx << std::endl;
//                    assert(false);
                }
            }

            int test = 0;
            if (At >= 0) test++;
            if (Bt >= 0) test += 2;
            if (Ct >= 0) test += 4;
            if (Dt >= 0) test += 8;
            if (test <= 4)
                return alongEdgeIdx > 0;
            else if (test == 5)
                if(At * Ct - Bt * Dt < FLT_EPSILON) return alongEdgeIdx > 0;
            else if (test == 6)
                return alongEdgeIdx > 0;
            else if (test == 7)
                return alongEdgeIdx < 0;
            else if (test <= 9)
                return alongEdgeIdx > 0;
            else if (test == 10)
                if(At * Ct - Bt * Dt >= FLT_EPSILON) return alongEdgeIdx > 0;
            else if (test == 11)
                return alongEdgeIdx < 0;
            else if (test == 12)
                return alongEdgeIdx > 0;
            else if (test <= 15)
                return alongEdgeIdx < 0;

            return alongEdgeIdx < 0;
        }


//传入的都需要是uint32_t，这里的caseIdx是自己定义的43种case的index


#define MAKE_HARD_CODE(caseIdx, triNum, configCaseIdx, subconfig13Value) ((triNum) | (caseIdx << 8) | (configCaseIdx << 16) | (subconfig13Value << 24))


        //(可选)0.统计每个voxel是否会生成三角形
        //1.得到每个cube对应的case index(43种)
        CUB_KERNEL void MCKernel0_ClassifyVoxelAndGenVertices(MCKernelParams params){
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

            if(x >= params.cu_mc_params.shape.x || y >= params.cu_mc_params.shape.y || z >= params.cu_mc_params.shape.z) return;

            const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
            const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
            const unsigned int load_count = (HashTableSize + thread_count - 1) / thread_count;
            const unsigned int thread_beg = thread_idx * load_count;
            const unsigned int thread_end = min(thread_beg + load_count, HashTableSize);

            __shared__ uint4 hash_table[HashTableSize][2];

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

            //全局的体素坐标
            uint3 voxel_coord = make_uint3(x, y, z) + params.cu_mc_params.origin;

            //采样得到一个cube对应的八个体素
            float field[8];
#define MAKE_XYZ(x, y, z) (( x << 2 ) | ( y << 1) | z)
            field[0] = VirtualSampling(params, hash_table, voxel_coord, params.cu_mc_params.lod, MAKE_XYZ(0, 0, 0)) - params.cu_mc_params.isovalue;
            field[1] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 0), params.cu_mc_params.lod, MAKE_XYZ(1, 0, 0)) - params.cu_mc_params.isovalue;
            field[2] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 0), params.cu_mc_params.lod, MAKE_XYZ(1, 1, 0)) - params.cu_mc_params.isovalue;
            field[3] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 0), params.cu_mc_params.lod, MAKE_XYZ(0, 1, 0)) - params.cu_mc_params.isovalue;
            field[4] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 0, 1), params.cu_mc_params.lod, MAKE_XYZ(0, 0, 1)) - params.cu_mc_params.isovalue;
            field[5] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 1), params.cu_mc_params.lod, MAKE_XYZ(1, 0, 1)) - params.cu_mc_params.isovalue;
            field[6] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 1), params.cu_mc_params.lod, MAKE_XYZ(1, 1, 1)) - params.cu_mc_params.isovalue;
            field[7] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 1), params.cu_mc_params.lod, MAKE_XYZ(0, 1, 1)) - params.cu_mc_params.isovalue;

            //计算索引，分类查表，计算出case idx(43种)，以及会生成的三角形数量，所有信息pack到一个uint32_t
            uint32_t config_index = 0;
//            config_index += uint32_t(field[0] < params.cu_mc_params.isovalue);
//            config_index += uint32_t(field[1] < params.cu_mc_params.isovalue) << 1;
//            config_index += uint32_t(field[2] < params.cu_mc_params.isovalue) << 2;
//            config_index += uint32_t(field[3] < params.cu_mc_params.isovalue) << 3;
//            config_index += uint32_t(field[4] < params.cu_mc_params.isovalue) << 4;
//            config_index += uint32_t(field[5] < params.cu_mc_params.isovalue) << 5;
//            config_index += uint32_t(field[6] < params.cu_mc_params.isovalue) << 6;
//            config_index += uint32_t(field[7] < params.cu_mc_params.isovalue) << 7;

            for(int i = 0; i < 8; i++){
                if(fabs(field[i]) < FLT_EPSILON) field[i] = FLT_EPSILON;
#ifdef USE_SDF
                if(field[i] > 0) config_index += 1u << i;
#else
                if(field[i] <= 0) config_index += 1u << i;
#endif
            }

            const unsigned int case_idx = cases[config_index][0];
            const unsigned int config_idx_in_case = cases[config_index][1];
            unsigned int subconfig = 0, subconfig13Value = 0;
            uint32_t m_code = 0, vert_num = 0;
            switch (case_idx) {
                // 0 : 生成 0 个三角形  内部 0 中情况
                case 0: {
                    m_code = MAKE_HARD_CODE(0, 0, config_idx_in_case, subconfig13Value);
                    vert_num = 0;
                    break;
                }

                // 1 : 生成 1 个三角形  内部 1 种情况
                case 1 : {
                    m_code = MAKE_HARD_CODE(1, 1, config_idx_in_case, subconfig13Value);
                    vert_num = 3;
                    break;
                }

                // 2 : 生成 2 个三角形  内部 1 种情况
                case 2 : {
                    m_code = MAKE_HARD_CODE(2, 2, config_idx_in_case, subconfig13Value);
                    vert_num = 6;
                    break;
                }
                // 3 : 内部 2 种情况  1:2  2:4
                case 3 : {
                    if(!TestFace(field, test3[config_idx_in_case])){
                        m_code = MAKE_HARD_CODE(3, 2, config_idx_in_case, subconfig13Value);
                        vert_num = 6;
                    }
                    else{
                        m_code = MAKE_HARD_CODE(4, 4, config_idx_in_case, subconfig13Value);
                        vert_num = 12;
                    }
                    break;
                }
                // 4 : 内部 2 种情况  1:2  2:6
                case 4 : {
                    if(TestInterior(field, case_idx, test4[config_idx_in_case], 1)){
                        m_code = MAKE_HARD_CODE(5, 2, config_idx_in_case, subconfig13Value);
                        vert_num = 6;
                    }
                    else{
                        m_code = MAKE_HARD_CODE(6, 6, config_idx_in_case, subconfig13Value);
                        vert_num = 18;
                    }
                    break;
                }
                // 5 : 生成 3 个三角形  内部 1 种情况
                case 5 : {
                    m_code = MAKE_HARD_CODE(7, 3, config_idx_in_case, subconfig13Value);
                    vert_num = 9;
                    break;
                }
                // 6 : 内部 3 种情况  1:3  2:9  3:5
                case 6 : {
                    if(TestFace(field, test6[config_idx_in_case][0]) == 0){
                        if(TestInterior(field, case_idx, test6[config_idx_in_case][1], test6[config_idx_in_case][2])){
                            m_code = MAKE_HARD_CODE(8, 3, config_idx_in_case, subconfig13Value);
                            vert_num = 9;
                        }
                        else{
                            m_code = MAKE_HARD_CODE(9, 9, config_idx_in_case, subconfig13Value);
                            vert_num = 27;
                        }
                    }
                    else{
                        m_code = MAKE_HARD_CODE(10, 5, config_idx_in_case, subconfig13Value);
                        vert_num = 15;
                    }
                    break;
                }
                // 7 : 内部 8 种情况  1:3  2:5  3:5  4:9  5:5  6:9  7:9  8:5  9:9
                case 7 : {
                    if(TestFace(field, test7[config_idx_in_case][0])){
                        subconfig += 1;
                    }
                    if(TestFace(field, test7[config_idx_in_case][1])){
                        subconfig += 2;
                    }
                    if(TestFace(field, test7[config_idx_in_case][2])){
                        subconfig += 4;
                    }
                    switch (subconfig) {
                        case 0 : {
                            m_code = MAKE_HARD_CODE(11, 3, config_idx_in_case, subconfig13Value);
                            vert_num = 9;
                            break;
                        }
                        case 1 : {
                            m_code = MAKE_HARD_CODE(12, 5, config_idx_in_case, subconfig13Value);
                            vert_num = 15;
                            break;
                        }
                        case 2 : {
                            m_code = MAKE_HARD_CODE(13, 5, config_idx_in_case, subconfig13Value);
                            vert_num = 15;
                            break;
                        }
                        case 3 : {
                            m_code = MAKE_HARD_CODE(14, 9, config_idx_in_case, subconfig13Value);
                            vert_num = 27;
                            break;
                        }
                        case 4 : {
                            m_code = MAKE_HARD_CODE(15, 5, config_idx_in_case, subconfig13Value);
                            vert_num = 15;
                            break;
                        }
                        case 5 : {
                            m_code = MAKE_HARD_CODE(16, 9, config_idx_in_case, subconfig13Value);
                            vert_num = 27;
                            break;
                        }
                        case 6 : {
                            m_code = MAKE_HARD_CODE(17, 9, config_idx_in_case, subconfig13Value);
                            vert_num = 27;
                            break;
                        }
                        case 7 : {
                            if(TestInterior(field, case_idx, test7[config_idx_in_case][3], test7[config_idx_in_case][4]) == 0){
                                m_code = MAKE_HARD_CODE(18, 5, config_idx_in_case, subconfig13Value);
                                vert_num = 15;
                            }
                            else{
                                m_code = MAKE_HARD_CODE(19, 9, config_idx_in_case, subconfig13Value);
                                vert_num = 27;
                            }
                            break;
                        }
                    }
                    break;
                }
                // 8 : 生成 2 个三角形  内部 1 种情况
                case 8 : {
                    m_code = MAKE_HARD_CODE(20, 2, config_idx_in_case, subconfig13Value);
                    vert_num = 6;
                    break;
                }
                // 9 : 生成 4 个三角形  内部 1 种情况
                case 9 : {
                    m_code = MAKE_HARD_CODE(21, 4, config_idx_in_case, subconfig13Value);
                    vert_num = 12;
                    break;
                }
                // 10 : 内部 5 种情况  1:4  2:8  3:8  4:8  5:4
                case 10 : {
                    if(TestFace(field, test10[config_idx_in_case][0])){
                        subconfig += 1;
                    }
                    if(TestFace(field, test10[config_idx_in_case][1])){
                        subconfig += 2;
                    }
                    switch (subconfig) {
                        case 0 : {
                            if(TestInterior(field, case_idx, test10[config_idx_in_case][2], 1)){
                                m_code = MAKE_HARD_CODE(22, 4, config_idx_in_case, subconfig13Value);
                                vert_num = 12;
                            }
                            else{
                                m_code = MAKE_HARD_CODE(23, 8, config_idx_in_case, subconfig13Value);
                                vert_num = 24;
                            }
                            break;
                        }
                        case 1 : {
                            m_code = MAKE_HARD_CODE(24, 8, config_idx_in_case, subconfig13Value);
                            vert_num = 24;
                            break;
                        }
                        case 2 : {
                            m_code = MAKE_HARD_CODE(25, 8, config_idx_in_case, subconfig13Value);
                            vert_num = 24;
                            break;
                        }
                        case 3 : {
                            m_code = MAKE_HARD_CODE(26, 4, config_idx_in_case, subconfig13Value);
                            vert_num = 12;
                            break;
                        }
                    }
                    break;
                }
                // 11 :生成 4 个三角形  内部 1 种情况
                case 11 : {
                    m_code = MAKE_HARD_CODE(27, 4, config_idx_in_case, subconfig13Value);
                    vert_num = 12;
                    break;
                }
                // 12 : 内部 5 种情况  1:4  2:8  3:8  4:8  5:4
                case 12 : {
                    if(TestFace(field, test12[config_idx_in_case][0])){
                        subconfig += 1;
                    }
                    if(TestFace(field, test12[config_idx_in_case][1])){
                        subconfig += 2;
                    }
                    switch (subconfig) {
                        case 0 : {
                            if(TestInterior(field, case_idx, test12[config_idx_in_case][2], test12[config_idx_in_case][3])){
                                m_code = MAKE_HARD_CODE(28, 4, config_idx_in_case, subconfig13Value);
                                vert_num = 12;
                            }
                            else{
                                m_code = MAKE_HARD_CODE(29, 8, config_idx_in_case, subconfig13Value);
                                vert_num = 24;
                            }
                            break;
                        }

                        case 1 : {
                            m_code = MAKE_HARD_CODE(30, 8, config_idx_in_case, subconfig13Value);
                            vert_num = 24;
                            break;
                        }

                        case 2 : {
                            m_code = MAKE_HARD_CODE(31, 8, config_idx_in_case, subconfig13Value);
                            vert_num = 24;
                            break;
                        }

                        case 3 : {
                            m_code = MAKE_HARD_CODE(32, 4, config_idx_in_case, subconfig13Value);
                            vert_num = 12;
                            break;
                        }
                    }
                    break;
                }
                // 13 : 内部 9 种情况  1:4  2:6  3:10  4:12  5:6  6:10  7:10  8:6  9:4
                case 13 : {
                    if(TestFace(field, test13[config_idx_in_case][0])){
                        subconfig += 1;
                    }
                    if(TestFace(field, test13[config_idx_in_case][1])){
                        subconfig += 2;
                    }
                    if(TestFace(field, test13[config_idx_in_case][2])){
                        subconfig += 4;
                    }
                    if(TestFace(field, test13[config_idx_in_case][3])){
                        subconfig += 8;
                    }
                    if(TestFace(field, test13[config_idx_in_case][4])){
                        subconfig += 16;
                    }
                    if(TestFace(field, test13[config_idx_in_case][5])){
                        subconfig += 32;
                    }
                    subconfig13Value = subconfig13[subconfig];
                    if(subconfig13Value == 0){
                        m_code = MAKE_HARD_CODE(33, 4, config_idx_in_case, subconfig13Value);
                        vert_num = 12;
                    }
                    else if(subconfig13Value < 7){
                        m_code = MAKE_HARD_CODE(34, 6, config_idx_in_case, subconfig13Value);
                        vert_num = 18;
                    }
                    else if(subconfig13Value < 19){
                        m_code = MAKE_HARD_CODE(35, 10, config_idx_in_case, subconfig13Value);
                        vert_num = 30;
                    }
                    else if(subconfig13Value < 23){
                        m_code = MAKE_HARD_CODE(36, 12, config_idx_in_case, subconfig13Value);
                        vert_num = 36;
                    }
                    else if(subconfig13Value < 27){
                        if(TestInterior(field, case_idx, test13[config_idx_in_case][6], tiling13_5_1[config_idx_in_case][subconfig13Value - 23][0])){
                            m_code = MAKE_HARD_CODE(37, 6, config_idx_in_case, subconfig13Value);
                            vert_num = 18;
                        }
                        else{
                            m_code = MAKE_HARD_CODE(38, 10, config_idx_in_case, subconfig13Value);
                            vert_num = 30;
                        }

                    }
                    else if(subconfig13Value < 39){
                        m_code = MAKE_HARD_CODE(39, 10, config_idx_in_case, subconfig13Value);
                        vert_num = 30;
                    }
                    else if(subconfig13Value < 45){
                        m_code = MAKE_HARD_CODE(40, 6, config_idx_in_case, subconfig13Value);
                        vert_num = 18;
                    }
                    else if(subconfig13Value == 45){
                        m_code = MAKE_HARD_CODE(41, 4, config_idx_in_case, subconfig13Value);
                        vert_num = 12;
                    }
                    break;
                }
                // 14 : 生成 4 个三角形  内部 1 种情况
                case 14 : {
                    m_code = MAKE_HARD_CODE(42, 4, config_idx_in_case, subconfig13Value);
                    vert_num = 12;
                    break;
                }
            }

            params.vol_code.at(x, y, z) = m_code;
            params.vertex_num.at(x, y, z) = vert_num;

#ifdef SKIP_EMPTY_VOXELS
            params.cube_occupied.at(x, y, z) = vert_num > 0 ? 1 : 0;
#endif
        }

        __forceinline__ CUB_GPU float3 VertexInterp(float isovalue, float3 p0, float3 p1, float f0, float f1){
            return lerp(p0, p1, (isovalue - f0) / (f1 - f0));
        }

        CUB_KERNEL void MCKernel_ClassifyVoxel(MCKernelParams params){
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

            if(x >= params.cu_mc_params.shape.x
                || y >= params.cu_mc_params.shape.y
                || z >= params.cu_mc_params.shape.z) return;
            if (params.vertex_num.at(x, y, z) > 0){
                size_t idx = z * params.cu_mc_params.shape.x * params.cu_mc_params.shape.y + y * params.cu_mc_params.shape.x + x;
                params.compacted_voxel_array.at(params.cube_occupied_scanned.at(idx)) = idx;
            }

        }

        CUB_KERNEL void MCKernel1_GenTriangles(MCKernelParams params){
      #ifdef SKIP_EMPTY_VOXELS
            uint32_t voxel_index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            if(voxel_index >= params.occupied_voxel_num) return;
            voxel_index = params.compacted_voxel_array.at(voxel_index);

            const unsigned z = voxel_index / (params.cu_mc_params.shape.x * params.cu_mc_params.shape.y);
            const unsigned y = (voxel_index % (params.cu_mc_params.shape.x * params.cu_mc_params.shape.y)) / params.cu_mc_params.shape.x;
            const unsigned x = voxel_index - y * params.cu_mc_params.shape.x
                            - z * params.cu_mc_params.shape.x * params.cu_mc_params.shape.y;
            uint3 voxel_coord = make_uint3(x, y, z) + params.cu_mc_params.origin;
      #else
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

            if(x >= params.cu_mc_params.shape.x
                || y >= params.cu_mc_params.shape.y
                || z >= params.cu_mc_params.shape.z)
                return;

            uint3 voxel_coord = make_uint3(x, y, z) + params.cu_mc_params.origin;
            uint32_t voxel_index = x + y * params.cu_mc_params.shape.x + z * params.cu_mc_params.shape.x * params.cu_mc_params.shape.y;
      #endif

            __syncthreads();

//            printf("ok1\n");

            const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
            const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
            const unsigned int load_count = (HashTableSize + thread_count - 1) / thread_count;
            const unsigned int thread_beg = thread_idx * load_count;
            const unsigned int thread_end = min(thread_beg + load_count, HashTableSize);

            // 32kb
            __shared__ uint4 hash_table[HashTableSize][2];

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




            float3 vert[8];
            vert[0] = make_float3( voxel_coord.x + 0.5f,       voxel_coord.y + 0.5f,       voxel_coord.z + 0.5f)      * params.cu_vol_params.space;
            vert[1] = make_float3((voxel_coord.x + 1) + 0.5f,  voxel_coord.y + 0.5f,       voxel_coord.z + 0.5f)      * params.cu_vol_params.space;
            vert[2] = make_float3((voxel_coord.x + 1) + 0.5f, (voxel_coord.y + 1) + 0.5f,  voxel_coord.z + 0.5f)      * params.cu_vol_params.space;
            vert[3] = make_float3( voxel_coord.x + 0.5f,      (voxel_coord.y + 1) + 0.5f,  voxel_coord.z + 0.5f)      * params.cu_vol_params.space;
            vert[4] = make_float3( voxel_coord.x + 0.5f,       voxel_coord.y + 0.5f,      (voxel_coord.z + 1) + 0.5f) * params.cu_vol_params.space;
            vert[5] = make_float3((voxel_coord.x + 1) + 0.5f,  voxel_coord.y + 0.5f,      (voxel_coord.z + 1) + 0.5f) * params.cu_vol_params.space;
            vert[6] = make_float3((voxel_coord.x + 1) + 0.5f, (voxel_coord.y + 1) + 0.5f, (voxel_coord.z + 1) + 0.5f) * params.cu_vol_params.space;
            vert[7] = make_float3( voxel_coord.x + 0.5f,      (voxel_coord.y + 1) + 0.5f, (voxel_coord.z + 1) + 0.5f) * params.cu_vol_params.space;


            float field[8];
#define MAKE_XYZ(x, y, z) (( x << 2 ) | ( y << 1) | z)
            field[0] = VirtualSampling(params, hash_table, voxel_coord, params.cu_mc_params.lod, MAKE_XYZ(0, 0, 0));
            field[1] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 0), params.cu_mc_params.lod, MAKE_XYZ(1, 0, 0));
            field[2] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 0), params.cu_mc_params.lod, MAKE_XYZ(1, 1, 0));
            field[3] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 0), params.cu_mc_params.lod, MAKE_XYZ(0, 1, 0));
            field[4] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 0, 1), params.cu_mc_params.lod, MAKE_XYZ(0, 0, 1));
            field[5] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 1), params.cu_mc_params.lod, MAKE_XYZ(1, 0, 1));
            field[6] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 1), params.cu_mc_params.lod, MAKE_XYZ(1, 1, 1));
            field[7] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 1), params.cu_mc_params.lod, MAKE_XYZ(0, 1, 1));

            //不需要生成法向量，因为之后会进行光滑操作，之后再合并重复顶点并生成索引，顶点法向量可以由面法向量插值得到，记录一个面所属的三个顶点，刚好与光滑的数据结构相同

            float3 vert_list[13];
            vert_list[0] = VertexInterp(params.cu_mc_params.isovalue, vert[0], vert[1], field[0], field[1]);
            vert_list[1] = VertexInterp(params.cu_mc_params.isovalue, vert[1], vert[2], field[1], field[2]);
            vert_list[2] = VertexInterp(params.cu_mc_params.isovalue, vert[3], vert[2], field[3], field[2]);
            vert_list[3] = VertexInterp(params.cu_mc_params.isovalue, vert[0], vert[3], field[0], field[3]);

            vert_list[4] = VertexInterp(params.cu_mc_params.isovalue, vert[4], vert[5], field[4], field[5]);
            vert_list[5] = VertexInterp(params.cu_mc_params.isovalue, vert[5], vert[6], field[5], field[6]);
            vert_list[6] = VertexInterp(params.cu_mc_params.isovalue, vert[7], vert[6], field[7], field[6]);
            vert_list[7] = VertexInterp(params.cu_mc_params.isovalue, vert[4], vert[7], field[4], field[7]);

            vert_list[8] = VertexInterp(params.cu_mc_params.isovalue, vert[0], vert[4], field[0], field[4]);
            vert_list[9] = VertexInterp(params.cu_mc_params.isovalue, vert[1], vert[5], field[1], field[5]);
            vert_list[10] = VertexInterp(params.cu_mc_params.isovalue, vert[2], vert[6], field[2], field[6]);
            vert_list[11] = VertexInterp(params.cu_mc_params.isovalue, vert[3], vert[7], field[3], field[7]);
            vert_list[12] = make_float3(0.f);
            uint32_t code = params.vol_code.at(x, y, z);

            const unsigned int m_case_idx = (code >> 8) & 0xff;
            const unsigned int tri_num = code & 0xff;
            const unsigned int config_idx_in_case = (code >> 16) & 0xff;
            const unsigned int subconfig13_val = (code >> 24) & 0xff;
            const char* edge_table = nullptr;

            if(tri_num == 0) return;
            __syncthreads();

            assert(tri_num == TriNumTable[m_case_idx]);
            assert(m_case_idx >= 0 && m_case_idx < 43);

            switch (m_case_idx){
                case 1 : edge_table = tiling1[config_idx_in_case]; break;
                case 2 : edge_table = tiling2[config_idx_in_case]; break;
                case 3 : edge_table = tiling3_1[config_idx_in_case]; break;
                case 4 : edge_table = tiling3_2[config_idx_in_case]; break;
                case 5 : edge_table = tiling4_1[config_idx_in_case]; break;
                case 6 : edge_table = tiling4_2[config_idx_in_case]; break;
                case 7 : edge_table = tiling5[config_idx_in_case]; break;
                case 8 : edge_table = tiling6_1_1[config_idx_in_case]; break;
                case 9 : edge_table = tiling6_1_2[config_idx_in_case]; break;
                case 10 : edge_table = tiling6_2[config_idx_in_case]; break;
                case 11 : edge_table = tiling7_1[config_idx_in_case]; break;
                case 12 : edge_table = m_tiling7_2[0][config_idx_in_case]; break;
                case 13 : edge_table = m_tiling7_2[1][config_idx_in_case]; break;
                case 14 : edge_table = m_tiling7_3[0][config_idx_in_case]; break;
                case 15 : edge_table = m_tiling7_2[2][config_idx_in_case]; break;
                case 16 : edge_table = m_tiling7_3[1][config_idx_in_case]; break;
                case 17 : edge_table = m_tiling7_3[2][config_idx_in_case]; break;
                case 18 : edge_table = tiling7_4_1[config_idx_in_case]; break;
                case 19 : edge_table = tiling7_4_2[config_idx_in_case]; break;
                case 20 : edge_table = tiling8[config_idx_in_case]; break;
                case 21 : edge_table = tiling9[config_idx_in_case]; break;
                case 22 : edge_table = tiling10_1_1[config_idx_in_case]; break;
                case 23 : edge_table = tiling10_1_2[config_idx_in_case]; break;
                case 24 : edge_table = tiling10_2[config_idx_in_case]; break;
                case 25 : edge_table = tiling10_2_[config_idx_in_case]; break;
                case 26 : edge_table = tiling10_1_1_[config_idx_in_case]; break;
                case 27 : edge_table = tiling11[config_idx_in_case]; break;
                case 28 : edge_table = tiling12_1_1[config_idx_in_case]; break;
                case 29 : edge_table = tiling12_1_2[config_idx_in_case]; break;
                case 30 : edge_table = tiling12_2[config_idx_in_case]; break;
                case 31 : edge_table = tiling12_2_[config_idx_in_case]; break;
                case 32 : edge_table = tiling12_1_1_[config_idx_in_case]; break;
                case 33 : edge_table = tiling13_1[config_idx_in_case]; break;
                case 34 : edge_table = tiling13_2[config_idx_in_case][subconfig13_val - 1]; break;
                case 35 : edge_table = tiling13_3[config_idx_in_case][subconfig13_val - 7]; break;
                case 36 : edge_table = tiling13_4[config_idx_in_case][subconfig13_val - 19]; break;
                case 37 : edge_table = tiling13_5_1[config_idx_in_case][subconfig13_val - 23]; break;
                case 38 : edge_table = tiling13_5_2[config_idx_in_case][subconfig13_val - 23]; break;
                case 39 : edge_table = tiling13_3_[config_idx_in_case][subconfig13_val - 27]; break;
                case 40 : edge_table = tiling13_2_[config_idx_in_case][subconfig13_val - 39]; break;
                case 41 : edge_table = tiling13_1_[config_idx_in_case]; break;
                case 42 : edge_table = tiling14[config_idx_in_case]; break;
            }

            bool computed12 = false;
            auto compute12 = [&](int idx){
                int cnt = 0;
                for(int i = 0; i < 4; i++){
                    if((field[i] - params.cu_mc_params.isovalue) * (field[(i + 1) % 4] - params.cu_mc_params.isovalue) < 0.f){
                        vert_list[12] += vert_list[i];
                        cnt += 1;
                    }
                }
                for(int i = 4; i < 8; i++){
                    if((field[i] - params.cu_mc_params.isovalue) * (field[(i + 1) % 4 + 4] - params.cu_mc_params.isovalue) < 0.f){
                        vert_list[12] += vert_list[i];
                        cnt += 1;
                    }
                }
                for(int i = 0; i < 4; i++){
                    if((field[i] - params.cu_mc_params.isovalue) * (field[i + 4] - params.cu_mc_params.isovalue) < 0.f){
                        vert_list[12] += vert_list[i + 8];
                        cnt += 1;
                    }
                }
                if(cnt)
                    vert_list[12] /= (float)cnt;
            };

            for(int i = 0; i < tri_num; i++){

                int a = edge_table[i * 3];
                int b = edge_table[i * 3 + 1];
                int c = edge_table[i * 3 + 2];

                //index表示这个体素之前已经有index个顶点生成
                uint32_t index = params.num_verts_scanned.at(voxel_index) + i * 3;

                if(index >= params.gen_vert_num){
                    assert(false);
                }

                if((a == 12 || b == 12 || c == 12) && !computed12){
                    compute12(index);
                    computed12 = true;
                }

                //check index
                if(index + 3 < params.max_vert_num){
                    params.vertex_pos.at(index) = vert_list[a];
                    params.vertex_pos.at(index + 1) = vert_list[b];
                    params.vertex_pos.at(index + 2) = vert_list[c];
                }
            }

        }

    }




    class MarchingCubeAlgoPrivate{
    public:
        CUDAContext ctx;

        CUDAStream compute_stream;

        MCKernelParams params;


        Handle<CUDABuffer> vol_mc_code;// max_voxel_num * sizeof(uint32_t)
        Handle<CUDABuffer> vol_vert_num;
        Handle<CUDABuffer> vol_cube_occupied;
        Handle<CUDABuffer> vol_mc_scanned;// max_voxel_num * sizeof(uint32_t)
        Handle<CUDABuffer> vol_cube_occupied_scanned;// max_voxel_num * sizeof(uint32_t)
        //一般体数据生成的三角形数量相对于体素数量是很少的
        //dev
        Handle<CUDABuffer> vol_vert_pos;// max_voxel_num / 8 * sizeof(float3)
        Handle<CUDABuffer> vol_compacted_voxel_array;// max_voxel_num / 8 * sizeof(uint32_t)
        //host
        Handle<CUDAHostBuffer> vol_gen_host_vert;


        size_t max_voxel_num;
        size_t max_vert_num;

        UnifiedRescUID uid;

        static UnifiedRescUID GenMCAlgoUID(){
            static std::atomic<size_t> g_uid = 1;
            auto uid = g_uid.fetch_add(1);
            return GenUnifiedRescUID(uid, UnifiedRescType::MCAlgo);
        }

        std::mutex mtx;
    };


    void MarchingCubeAlgo::Lock() {
        _->mtx.lock();
    }


    void MarchingCubeAlgo::UnLock() {
        _->mtx.unlock();
    }


    UnifiedRescUID MarchingCubeAlgo::GetUID() const {
        return _->uid;
    }


    MarchingCubeAlgo::MarchingCubeAlgo(const MarchingCubeAlgoCreateInfo& info) {
        _ = std::make_unique<MarchingCubeAlgoPrivate>();

        _->ctx = info.gpu_mem_mgr._get_ptr()->_get_cuda_context();
        //使用null流，如果和渲染同一个ctx，那么两者不能并行，即进行mc的时候不能渲染
        _->compute_stream = cub::cu_stream::null(info.gpu_mem_mgr._get_ptr()->_get_cuda_context());

        _->max_voxel_num = info.max_voxel_num;
        _->max_vert_num = info.max_voxel_num / 8;

        _->vol_mc_code = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num * sizeof(uint32_t));
        _->vol_mc_scanned = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num * sizeof(uint32_t));
        _->vol_vert_num = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num * sizeof(uint32_t));

        _->vol_gen_host_vert = info.host_mem_mgr.Invoke(&HostMemMgr::AllocPinnedHostMem, ResourceType::Buffer, info.max_voxel_num / 8 * sizeof(Float3), false);
        _->vol_vert_pos = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num / 8 * sizeof(Float3));
        _->params.vertex_pos = _->vol_vert_pos->view_1d<float3>(_->max_vert_num);
#ifdef SKIP_EMPTY_VOXELS
        _->vol_cube_occupied = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num * sizeof(uint32_t));
        _->vol_compacted_voxel_array = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num / 8 * sizeof(uint32_t));
        _->vol_cube_occupied_scanned = info.gpu_mem_mgr.Invoke(&GPUMemMgr::AllocBuffer, ResourceType::Buffer, info.max_voxel_num * sizeof(uint32_t));
        _->params.compacted_voxel_array = _->vol_compacted_voxel_array->view_1d<uint32_t>(_->max_voxel_num / 8);
#endif
        _->params.max_vert_num = _->max_vert_num;

        _->uid = _->GenMCAlgoUID();
    }


    MarchingCubeAlgo::~MarchingCubeAlgo() {

    }

    struct BOP{
        CUB_GPU uint32_t operator()(uint32_t a, uint32_t b) const {
            return a + b;
        }
    };

    int MarchingCubeAlgo::Run(MarchingCubeAlgoParams &mc_params) {
        //一个线程复杂一个cube，一个block共用page table
        //一个block的线程数不要太多，因为之后可能要创建的shared memory大小和线程数有关，或者每个线程的寄存器文件太多
        //具体的线程数，之后可以再调整看看，找到一个tradeoff
        const dim3 tile = {4u, 4u, 4u};
        const dim3 grid_dim = {
                (mc_params.shape.x + tile.x - 1) / tile.x,
                (mc_params.shape.y + tile.y - 1) / tile.y,
                (mc_params.shape.z + tile.z - 1) / tile.z
        };
        cub::cu_kernel_launch_info info{grid_dim, tile};

        size_t num_voxels = (size_t)mc_params.shape.x * mc_params.shape.y * mc_params.shape.z;

        _->params.cu_mc_params.isovalue = mc_params.isovalue;
        _->params.cu_mc_params.lod = mc_params.lod;
        _->params.cu_mc_params.origin = {
                mc_params.origin.x, mc_params.origin.y, mc_params.origin.z
        };
        _->params.cu_mc_params.shape = {
                mc_params.shape.x, mc_params.shape.y, mc_params.shape.z
        };

        _->params.vol_code = _->vol_mc_code->view_3d<uint32_t>(cub::pitched_buffer_info{mc_params.shape.x * sizeof(uint32_t)},
                                                               cub::cu_extent{mc_params.shape.x, mc_params.shape.y, mc_params.shape.z});
        _->params.vertex_num = _->vol_vert_num->view_3d<uint32_t>(cub::pitched_buffer_info{mc_params.shape.x * sizeof(uint32_t)},
                                                                  cub::cu_extent{mc_params.shape.x, mc_params.shape.y, mc_params.shape.z});
        _->params.num_verts_scanned = _->vol_mc_scanned->view_1d<uint32_t>(num_voxels);
#ifdef SKIP_EMPTY_VOXELS
        _->params.cube_occupied = _->vol_cube_occupied->view_3d<uint32_t>(cub::pitched_buffer_info{mc_params.shape.x * sizeof(uint32_t)},
                                                                          cub::cu_extent{mc_params.shape.x, mc_params.shape.y, mc_params.shape.z});
        _->params.cube_occupied_scanned = _->vol_cube_occupied_scanned->view_1d<uint32_t>(num_voxels);
#endif

        assert(num_voxels > 0);
        if(num_voxels > _->max_voxel_num){
            throw std::runtime_error("Too many voxels for cuda marching cube algo!!!");
        }

        try{
            void* params[] = {&_->params};

            cub::cu_kernel::pending(info, &MCKernel0_ClassifyVoxelAndGenVertices, params)
            .launch(_->compute_stream).check_error_on_throw();

            // view as 1d to scan
            auto vol_mc_vert_num_view = _->vol_vert_num->view_1d<uint32_t>(num_voxels);
            auto vol_mc_scanned_view = _->vol_mc_scanned->view_1d<uint32_t>(num_voxels);

            //exclusive 结果第一个元素为 0
            thrust::exclusive_scan(thrust::device_ptr<uint32_t>(vol_mc_vert_num_view.data()), thrust::device_ptr<uint32_t>(vol_mc_vert_num_view.data() + num_voxels),
                    thrust::device_ptr<uint32_t>(vol_mc_scanned_view.data()), 0, BOP{});


            //计算生成的三角形数
            uint32_t last_ele, last_scan_ele;
            CUB_CHECK(cudaMemcpy(&last_ele, &vol_mc_vert_num_view.at(num_voxels - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUB_CHECK(cudaMemcpy(&last_scan_ele, &vol_mc_scanned_view.at(num_voxels - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));

            uint32_t total_vert_num = last_ele + last_scan_ele;
            _->params.gen_vert_num = total_vert_num;

#ifdef SKIP_EMPTY_VOXELS
            auto vol_cube_occupied_view = _->vol_cube_occupied->view_1d<uint32_t>(num_voxels);
            auto vol_cube_occupied_scanned_view = _->vol_cube_occupied_scanned->view_1d<uint32_t>(num_voxels);
            thrust::exclusive_scan(thrust::device_ptr<uint32_t>(vol_cube_occupied_view.data()), thrust::device_ptr<uint32_t>(vol_cube_occupied_view.data() + num_voxels),
                                   thrust::device_ptr<uint32_t>(vol_cube_occupied_scanned_view.data()), 0, BOP{});


            // 实际生成三角形的体素数量
            CUB_CHECK(cudaMemcpy(&last_ele, &vol_cube_occupied_view.at(num_voxels - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUB_CHECK(cudaMemcpy(&last_scan_ele, &vol_cube_occupied_scanned_view.at(num_voxels - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));

            uint32_t total_occupied_voxel_num = last_ele + last_scan_ele;
            LOG_DEBUG("total occupied voxel num : {}", total_occupied_voxel_num);
            _->params.occupied_voxel_num = total_occupied_voxel_num;

            cub::cu_kernel::pending(info, &MCKernel_ClassifyVoxel, params)
                .launch(_->compute_stream).check_error_on_throw();

            info.block_dim = dim3(32, 1, 1);
            info.grid_dim = dim3((total_occupied_voxel_num + 31) / 32, 1, 1);
            if(info.grid_dim.x > 65535u){
                info.grid_dim.y = info.grid_dim.x / 32768u;
                info.grid_dim.x = 32768u;
            }
#endif

            if(total_vert_num > 0){
                cub::cu_kernel::pending(info, &MCKernel1_GenTriangles, params)
                    .launch(_->compute_stream)
                    .check_error_on_throw();

                // 将结果从dev传回host
                auto dev_ret = _->vol_vert_pos->view_1d<Float3>(total_vert_num);
                auto host_ret = _->vol_gen_host_vert->view_1d<Float3>(total_vert_num);
                cub::memory_transfer_info tf_info;
                tf_info.width_bytes = total_vert_num * sizeof(Float3);

                cub::cu_memory_transfer(dev_ret, host_ret, tf_info).launch(_->compute_stream).check_error_on_throw();

                mc_params.gen_host_vertices_ret = host_ret;
            }

            return total_vert_num / 3;
        }
        catch (const std::exception& err) {
            LOG_ERROR("{}", err.what());
            return 0;
        }
    }

    void MarchingCubeAlgo::BindVTexture(VTextureHandle handle, TextureUnit unit) {
        assert(unit >= 0 && unit < MaxCUDATextureCountPerGPU);
#ifdef USE_SDF
        _->params.cu_vtex[unit] = handle->view_as({
            cub::e_clamp, cub::e_nearest, cub::e_raw, false
        });
#else
        _->params.cu_vtex[unit] = handle->view_as({
            cub::e_clamp, cub::e_nearest, cub::e_normalized_float, false
        });
#endif
    }

    void MarchingCubeAlgo::BindPTBuffer(PTBufferHandle handle) {
        _->params.cu_page_table.table = handle->view_1d<HashTableItem>(HashTableSize * sizeof(HashTableItem));
    }

    void MarchingCubeAlgo::SetVolume(const VolumeParams &volume_params) {
            _->params.cu_vol_params.space = {
                    volume_params.space.x,
                    volume_params.space.y,
                    volume_params.space.z
            };
            _->params.cu_vol_params.block_length = volume_params.block_length;
            _->params.cu_vol_params.padding = volume_params.padding;
            _->params.cu_vol_params.voxel_dim = {
                    volume_params.voxel_dim.x,
                    volume_params.voxel_dim.y,
                    volume_params.voxel_dim.z
            };
    }

#ifdef USE_LINEAR_BUFFER_FOR_TEXTURE
    void MarchingCubeAlgo::BindVBuffer(CUDABufferView3D<uint8_t> view, TextureUnit unit)
    {
        assert(unit >= 0 && unit < MaxCUDATextureCountPerGPU);
        _->params.cu_vbuf[unit] = view;
    }

#endif

    VISER_END
