#undef UTIL_ENABLE_OPENGL

#include <Algorithm/MarchingCube.hpp>
#include "LookUpTable.cuh"
#include "../Common/helper_math.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

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
            cudaTextureObject_t cu_vtex[MaxCUDATextureCountPerGPU];
            CUDAPageTable cu_page_table;
            CUDAVolumeParams cu_vol_params;
            CUDAMCAlgoParams cu_mc_params;

            CUDABufferView3D<uint32_t> vol_code;
            CUDABufferView3D<uint32_t> vertex_num;
            // 43 种情况对应的edge table起始地址
            CUDABufferView1D<float3> vertex_pos;
            uint32_t gen_vert_num;
            //存储每一个体素之前总共产生的顶点数量
            CUDABufferView1D<uint32_t> num_verts_scanned;
            uint32_t max_vert_num;

        };

        CUB_GPU float VirtualSampling(const MCKernelParams& params,
                              uint4 hash_table[][2],
                              uint3 voxel_coord, uint32_t lod){
            uint32_t lod_block_length = params.cu_vol_params.block_length << lod;
            uint3 block_uid = voxel_coord / lod_block_length;
            uint3 offset_in_block = (voxel_coord - block_uid * lod_block_length) / (1 << lod);
            uint4 key = make_uint4(block_uid, lod | (VolumeBlock_IsSWC << 8));
            uint4 tex_coord = Query(key, hash_table);
            uint32_t tid = (tex_coord.w >> 16) & 0xffff;
            uint3 coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
            float ret;
            if((tex_coord.w & 0xffff) & TexCoordFlag_IsValid){
                uint32_t block_size = params.cu_vol_params.block_length + params.cu_vol_params.padding * 2;
                auto pos = coord * block_size + offset_in_block + params.cu_vol_params.padding;
                ret = tex3D<float>(params.cu_vtex[tid], pos.x, pos.y, pos.z);
            }
            else{
                ret = 0.f;
            }
            return ret;
        }


        CUB_GPU float TestFace(const float field[8], int f){
            // 渐近线测试所用的四个参数
            int A, B, C, D;
            // 注意有的面需要将结果取反，这个时候 f 是负数
            // 参考 Figure 6 第 3 个图，这个顺序 ABCD 的应该是作者自己确定的，论文里面没写，这个顺序跟测试是否带正负号相关
            switch (f) {
                case 1:
                case -1:
                    A = 0, B = 4, C = 5, D = 1;
                    break;
                case 2:
                case -2:
                    A = 1, B = 5, C = 6, D = 2;
                    break;
                case 3:
                case -3:
                    A = 2, B = 6, C = 7, D = 3;
                    break;
                case 4:
                case -4:
                    A = 3, B = 7, C = 4, D = 0;
                    break;
                case 5:
                case -5:
                    A = 0, B = 3, C = 2, D = 1;
                    break;
                case 6:
                case -6:
                    A = 4, B = 7, C = 6, D = 5;
                    break;
                default:{

                }
            }

            return f * field[A] * (field[A] * field[C] - field[B] * field[D]);
        }

        CUB_GPU float TestInterior(const float field[8], int caseIdx, int alongEdgeIdx, int edgeIdx){
            float t, At = 0, Bt = 0, Ct = 0, Dt = 0, a, b;
            switch (caseIdx) {
                // 强对称性，直接计算
                case 4:
                case 10:
                    a = (field[4] - field[0]) * (field[6] - field[2]) - (field[7] - field[3]) * (field[5] - field[1]);
                    b = field[2] * (field[4] - field[0]) + field[0] * (field[6] - field[2]) - field[1] * (field[7] - field[3]) - field[3] * (field[5] - field[1]);
                    t = -b / (2 * a);
                    if (t > 0 || t < 1) return -alongEdgeIdx;
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
                return -alongEdgeIdx;
            else if (test == 5)
                return (At * Ct - Bt * Dt) * alongEdgeIdx;
            else if (test == 6)
                return -alongEdgeIdx;
            else if (test == 7)
                return alongEdgeIdx;
            else if (test <= 9)
                return -alongEdgeIdx;
            else if (test == 10)
                return -(At * Ct - Bt * Dt) * alongEdgeIdx;
            else if (test == 11)
                return alongEdgeIdx;
            else if (test == 12)
                return -alongEdgeIdx;
            else if (test <= 15)
                return alongEdgeIdx;
            else {
//                std::cout << "testInterior got wrong test value: " << test << std::endl;
//                assert(false);
                return -1;
            }
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
            field[0] = VirtualSampling(params, hash_table, voxel_coord, params.cu_mc_params.lod);
            field[1] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 0), params.cu_mc_params.lod);
            field[2] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 0), params.cu_mc_params.lod);
            field[3] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 0), params.cu_mc_params.lod);
            field[4] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 0, 1), params.cu_mc_params.lod);
            field[5] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 1), params.cu_mc_params.lod);
            field[6] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 1), params.cu_mc_params.lod);
            field[7] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 1), params.cu_mc_params.lod);

            //计算索引，分类查表，计算出case idx(43种)，以及会生成的三角形数量，所有信息pack到一个uint32_t
            uint32_t config_index = 0;
            config_index += uint32_t(field[0] < params.cu_mc_params.isovalue);
            config_index += uint32_t(field[1] < params.cu_mc_params.isovalue) << 1;
            config_index += uint32_t(field[2] < params.cu_mc_params.isovalue) << 2;
            config_index += uint32_t(field[3] < params.cu_mc_params.isovalue) << 3;
            config_index += uint32_t(field[4] < params.cu_mc_params.isovalue) << 4;
            config_index += uint32_t(field[5] < params.cu_mc_params.isovalue) << 5;
            config_index += uint32_t(field[6] < params.cu_mc_params.isovalue) << 6;
            config_index += uint32_t(field[7] < params.cu_mc_params.isovalue) << 7;

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
                    if(TestFace(field, test3[config_idx_in_case]) < 0){
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
                    if(TestInterior(field, case_idx, test4[config_idx_in_case], 1) < 0){
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
                    if(TestFace(field, test6[config_idx_in_case][0]) < 0){
                        if(TestInterior(field, case_idx, test6[config_idx_in_case][1], test6[config_idx_in_case][2]) < 0){
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
                    if(TestFace(field, test7[config_idx_in_case][0]) < 0){
                        subconfig += 1;
                    }
                    if(TestFace(field, test7[config_idx_in_case][1]) < 0){
                        subconfig += 2;
                    }
                    if(TestFace(field, test7[config_idx_in_case][2]) < 0){
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
                            if(TestInterior(field, case_idx, test7[config_idx_in_case][3], test7[config_idx_in_case][4]) > 0){
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
                    if(TestFace(field, test10[config_idx_in_case][0]) > 0){
                        subconfig += 1;
                    }
                    if(TestFace(field, test10[config_idx_in_case][1]) > 0){
                        subconfig += 2;
                    }
                    switch (subconfig) {
                        case 0 : {
                            if(TestInterior(field, case_idx, test10[config_idx_in_case][2], 1) < 0){
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
                    if(TestFace(field, test12[config_idx_in_case][0]) > 0){
                        subconfig += 1;
                    }
                    if(TestFace(field, test12[config_idx_in_case][1]) > 0){
                        subconfig += 2;
                    }
                    switch (subconfig) {
                        case 0 : {
                            if(TestInterior(field, case_idx, test12[config_idx_in_case][2], test12[config_idx_in_case][3]) < 0){
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
                    if(TestFace(field, test13[config_idx_in_case][0]) > 0){
                        subconfig += 1;
                    }
                    if(TestFace(field, test13[config_idx_in_case][1]) > 0){
                        subconfig += 2;
                    }
                    if(TestFace(field, test13[config_idx_in_case][2]) > 0){
                        subconfig += 4;
                    }
                    if(TestFace(field, test13[config_idx_in_case][3]) > 0){
                        subconfig += 8;
                    }
                    if(TestFace(field, test13[config_idx_in_case][4]) > 0){
                        subconfig += 16;
                    }
                    if(TestFace(field, test13[config_idx_in_case][5]) > 0){
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
                        if(TestInterior(field, case_idx, test13[config_idx_in_case][6], tiling13_5_1[config_idx_in_case][subconfig13Value - 23][0]) < 0){
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
        }

        __forceinline__ CUB_GPU float3 VertexInterp(float isovalue, float3 p0, float3 p1, float f0, float f1){
            return lerp(p0, p1, (isovalue - f0) / (f1 - f0));
        }

        CUB_KERNEL void MCKernel1_GenTriangles(MCKernelParams params){
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

            if(x >= params.cu_mc_params.shape.x
               || y >= params.cu_mc_params.shape.y
               || z >= params.cu_mc_params.shape.z)
                return;

            __shared__ const char*const* tri_edge_table[43];
            __shared__ const char*const*const* case13_edge_table[7];
            __shared__ int case13_offset[7];
            if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
                tri_edge_table[0] = nullptr; // 0
                tri_edge_table[1] = reinterpret_cast<const char*const*>(tiling1); // 1
                tri_edge_table[2] = reinterpret_cast<const char*const*>(tiling2); // 2
                tri_edge_table[3] = reinterpret_cast<const char*const*>(tiling3_1); // 3.1
                tri_edge_table[4] = reinterpret_cast<const char*const*>(tiling3_2); // 3.2
                tri_edge_table[5] = reinterpret_cast<const char*const*>(tiling4_1); // 4.1
                tri_edge_table[6] = reinterpret_cast<const char*const*>(tiling4_2); // 4.2
                tri_edge_table[7] = reinterpret_cast<const char*const*>(tiling5); // 5
                tri_edge_table[8] = reinterpret_cast<const char*const*>(tiling6_1_1); // 6.1.1 or 6.1
                tri_edge_table[9] = reinterpret_cast<const char*const*>(tiling6_1_2); // 6.1.2 or 6.2
                tri_edge_table[10] = reinterpret_cast<const char*const*>(tiling6_2); // 6.2 or 6.3
                tri_edge_table[11] = reinterpret_cast<const char*const*>(tiling7_1); // 7.1
                tri_edge_table[12] = reinterpret_cast<const char*const*>(m_tiling7_2[0]); // 7.2
                tri_edge_table[13] = reinterpret_cast<const char*const*>(m_tiling7_2[1]); // 7.2 or 7.3
                tri_edge_table[14] = reinterpret_cast<const char*const*>(m_tiling7_3[0]); // 7.3 or 7.4
                tri_edge_table[15] = reinterpret_cast<const char*const*>(m_tiling7_2[2]); // 7.3 or 7.5
                tri_edge_table[16] = reinterpret_cast<const char*const*>(m_tiling7_3[1]); // 7.3 or 7.6
                tri_edge_table[17] = reinterpret_cast<const char*const*>(m_tiling7_3[2]); // 7.3 or 7.7
                tri_edge_table[18] = reinterpret_cast<const char*const*>(tiling7_4_1); // 7.4.1 or 7.8
                tri_edge_table[19] = reinterpret_cast<const char*const*>(tiling7_4_2); // 7.4.2 or 7.9
                tri_edge_table[20] = reinterpret_cast<const char*const*>(tiling8); // 8
                tri_edge_table[21] = reinterpret_cast<const char*const*>(tiling9); // 9
                tri_edge_table[22] = reinterpret_cast<const char*const*>(tiling10_1_1); // 10.1.1 or 10.1
                tri_edge_table[23] = reinterpret_cast<const char*const*>(tiling10_1_2); // 10.1.2 or 10.2
                tri_edge_table[24] = reinterpret_cast<const char*const*>(tiling10_2); // 10.2 or 10.3
                tri_edge_table[25] = reinterpret_cast<const char*const*>(tiling10_2_); // 10.2 or 10.4
                tri_edge_table[26] = reinterpret_cast<const char*const*>(tiling10_1_1_); // 10.1.1 or 10.5
                tri_edge_table[27] = reinterpret_cast<const char*const*>(tiling11); // 11
                tri_edge_table[28] = reinterpret_cast<const char*const*>(tiling12_1_1); // 12.1.1 or 12.1
                tri_edge_table[29] = reinterpret_cast<const char*const*>(tiling12_1_2); // 12.1.2 or 12.2
                tri_edge_table[30] = reinterpret_cast<const char*const*>(tiling12_2); // 12.2 or 12.3
                tri_edge_table[31] = reinterpret_cast<const char*const*>(tiling12_2_); // 12.2 or 12.4
                tri_edge_table[32] = reinterpret_cast<const char*const*>(tiling12_1_1_); // 12.1.1 or 12.5
                tri_edge_table[33] = reinterpret_cast<const char*const*>(tiling13_1); // 13.1
                case13_edge_table[0] = reinterpret_cast<const char*const*const*>(tiling13_2); // 13.2
                case13_offset[0] = 1;
                case13_edge_table[1] = reinterpret_cast<const char*const*const*>(tiling13_3); // 13.3
                case13_offset[1] = 7;
                case13_edge_table[2] = reinterpret_cast<const char*const*const*>(tiling13_4); // 13.4
                case13_offset[2] = 19;
                case13_edge_table[3] = reinterpret_cast<const char*const*const*>(tiling13_5_1); // 13.5.1 or 13.5
                case13_offset[3] = 23;
                case13_edge_table[4] = reinterpret_cast<const char*const*const*>(tiling13_5_2); // 13.5.2 or 13.6
                case13_offset[4] = 23;
                case13_edge_table[5] = reinterpret_cast<const char*const*const*>(tiling13_3_); // 13.3 or 13.7
                case13_offset[5] = 27;
                case13_edge_table[6] = reinterpret_cast<const char*const*const*>(tiling13_2_); // 13.2 or 13.8
                case13_offset[6] = 39;
                tri_edge_table[41] = reinterpret_cast<const char*const*>(tiling13_1_); // 13.1 or 13.9
                tri_edge_table[42] = reinterpret_cast<const char*const*>(tiling14); // 14

                if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
                    for(int i = 0; i < 34; i++){
                        printf("tri_edge_table[%d]: %lld\n", i, (size_t)tri_edge_table[i]);
                    }
                }
            }

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


            uint3 voxel_coord = make_uint3(x, y, z) + params.cu_mc_params.origin;
            uint32_t voxel_index = x + y * params.cu_mc_params.shape.x + z * params.cu_mc_params.shape.x * params.cu_mc_params.shape.y;

            float3 p = make_float3(voxel_coord.x + 0.5f, voxel_coord.y + 0.5f, voxel_coord.z + 0.5f) * params.cu_vol_params.space;
            float3 vert[8];
            vert[0] = p;
            vert[1] = p + make_float3(params.cu_vol_params.space.x, 0.f, 0.f);
            vert[2] = p + make_float3(params.cu_vol_params.space.x, params.cu_vol_params.space.y, 0.f);
            vert[3] = p + make_float3(0.f, params.cu_vol_params.space.y, 0.f);
            vert[4] = p + make_float3(0.f, 0.f, params.cu_vol_params.space.z);
            vert[5] = p + make_float3(params.cu_vol_params.space.x, 0.f, params.cu_vol_params.space.z);
            vert[6] = p + make_float3(params.cu_vol_params.space.x, params.cu_vol_params.space.y, params.cu_vol_params.space.z);
            vert[7] = p + make_float3(0.f, params.cu_vol_params.space.y, params.cu_vol_params.space.z);


            float field[8];
            field[0] = VirtualSampling(params, hash_table, voxel_coord, params.cu_mc_params.lod);
            field[1] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 0), params.cu_mc_params.lod);
            field[2] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 0), params.cu_mc_params.lod);
            field[3] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 0), params.cu_mc_params.lod);
            field[4] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 0, 1), params.cu_mc_params.lod);
            field[5] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 1), params.cu_mc_params.lod);
            field[6] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 1), params.cu_mc_params.lod);
            field[7] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 1), params.cu_mc_params.lod);

            //不需要生成法向量，因为之后会进行光滑操作，之后再合并重复顶点并生成索引，顶点法向量可以由面法向量插值得到，记录一个面所属的三个顶点，刚好与光滑的数据结构相同

            // 12 kb
//            __shared__ float3 vert_list[12 * ThreadsPerBlocks];
            float3 vert_list[12];
            vert_list[0] = VertexInterp(params.cu_mc_params.isovalue, vert[0], vert[1], field[0], field[1]);
            vert_list[1] = VertexInterp(params.cu_mc_params.isovalue, vert[1], vert[2], field[1], field[2]);
            vert_list[2] = VertexInterp(params.cu_mc_params.isovalue, vert[2], vert[3], field[2], field[3]);
            vert_list[3] = VertexInterp(params.cu_mc_params.isovalue, vert[3], vert[0], field[3], field[0]);

            vert_list[4] = VertexInterp(params.cu_mc_params.isovalue, vert[4], vert[5], field[4], field[5]);
            vert_list[5] = VertexInterp(params.cu_mc_params.isovalue, vert[5], vert[6], field[5], field[6]);
            vert_list[6] = VertexInterp(params.cu_mc_params.isovalue, vert[6], vert[7], field[6], field[7]);
            vert_list[7] = VertexInterp(params.cu_mc_params.isovalue, vert[7], vert[4], field[7], field[4]);

            vert_list[8] = VertexInterp(params.cu_mc_params.isovalue, vert[0], vert[4], field[0], field[4]);
            vert_list[9] = VertexInterp(params.cu_mc_params.isovalue, vert[1], vert[5], field[1], field[5]);
            vert_list[10] = VertexInterp(params.cu_mc_params.isovalue, vert[2], vert[6], field[2], field[6]);
            vert_list[11] = VertexInterp(params.cu_mc_params.isovalue, vert[3], vert[7], field[3], field[7]);

            uint32_t code = params.vol_code.at(x, y, z);

//            printf("ok2\n");

            const unsigned int m_case_idx = (code >> 8) & 0xff;
            const unsigned int tri_num = code & 0xff;
            const unsigned int config_idx_in_case = (code >> 16) & 0xff;
            const unsigned int subconfig13_val = (code >> 24) & 0xff;
            const char* edge_table = nullptr;
            if(tri_num == 0) return;
            __syncthreads();
//            printf("tri_num %u\n", tri_num);
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
                case 34 : edge_table = tiling13_2[config_idx_in_case][subconfig13_val - 1];
                case 35 : edge_table = tiling13_3[config_idx_in_case][subconfig13_val - 7];
                case 36 : edge_table = tiling13_4[config_idx_in_case][subconfig13_val - 19];
                case 37 : edge_table = tiling13_5_1[config_idx_in_case][subconfig13_val - 23];
                case 38 : edge_table = tiling13_5_2[config_idx_in_case][subconfig13_val - 23];
                case 39 : edge_table = tiling13_3_[config_idx_in_case][subconfig13_val - 27];
                case 40 : edge_table = tiling13_2_[config_idx_in_case][subconfig13_val - 39];
                case 41 : edge_table = tiling13_1_[config_idx_in_case];
                case 42 : edge_table = tiling14[config_idx_in_case];
            }

//            if(m_case_idx < 34 || m_case_idx > 40){
////                printf("m_case_idx %u, config_idx_in_case %u\n", m_case_idx, config_idx_in_case);
//                printf("okok %lld, m_case_idx %u, config_idx_in_case %u\n",
//                       (size_t)tri_edge_table[m_case_idx], m_case_idx, config_idx_in_case);
//                assert(config_idx_in_case < MyCaseTable[m_case_idx]);
//                edge_table = tri_edge_table[m_case_idx][config_idx_in_case];
//                printf("okok2 %lld, m_case_idx %u, config_idx_in_case %u, edge_table %lld\n",
//                       (size_t)tri_edge_table[m_case_idx], m_case_idx, config_idx_in_case,
//                       (size_t)edge_table);
//            }
//            else{
////                printf("m_case_idx %u, config_idx_in_case %u, subconfig13_val %u\n",
////                       m_case_idx, config_idx_in_case, subconfig13_val);
//                printf("exit\n");
//                return;
//                edge_table = case13_edge_table[m_case_idx - 34][config_idx_in_case][subconfig13_val - case13_offset[m_case_idx - 34]];
//            }


//            int a = edge_table[0 * 3];
//            int b = edge_table[0 * 3 + 1];
//            int c = edge_table[0 * 3 + 2];
//            printf("ok3 edge table %lld a %d b %d c %d\n", edge_table, a, b, c);
//            return;
            for(int i = 0; i < tri_num; i++){

                int a = edge_table[i * 3];
                int b = edge_table[i * 3 + 1];
                int c = edge_table[i * 3 + 2];

                //index表示这个体素之前已经有index个顶点生成
                uint32_t index = params.num_verts_scanned.at(voxel_index) + i * 3;
//                printf("tri_num %d i %d, a %d b %d c %d voxel_index %d index %d vert_num %d\n",
//                       tri_num, i, a, b, c, voxel_index, index,
//                       params.vertex_num.at(x, y, z));
                if(index >= params.gen_vert_num){
//                    printf("index %d, gen_vert_num %d, tri_num %d, m_case_idx %d,"
//                           "vert_num %d\n", index, params.gen_vert_num, tri_num,
//                           m_case_idx, params.vertex_num.at(x, y, z));
                    assert(false);
                }
                //check index
                if(index + 3 < params.max_vert_num){
                    params.vertex_pos.at(index) = vert_list[a];
                    params.vertex_pos.at(index + 1) = vert_list[b];
                    params.vertex_pos.at(index + 2) = vert_list[c];
                    printf("m_case_idx %d, gen tri index : %d, vert a %d: %f %f %f, vert b %d: %f %f %f, vert c %d: %f %f %f\n"
                           "filed0: %f, filed1: %f, field2: %f, filed3: %f\n"
                           "filed4: %f, filed5: %f, field6: %f, filed7: %f\n",
                           m_case_idx, index,
                           a, vert_list[a].x, vert_list[a].y, vert_list[c].z,
                           b, vert_list[b].x, vert_list[b].y, vert_list[b].z,
                           c, vert_list[c].x, vert_list[c].y, vert_list[c].z,
                           field[0], field[1], field[2], field[3], field[4], field[5], field[6], field[7]
                           );
                }
            }

        }

    }




    class MarchingCubeAlgoPrivate{
    public:
        cub::cu_context ctx;

        cub::cu_stream compute_stream;

        MCKernelParams params;


        Handle<CUDABuffer> vol_mc_code;// max_voxel_num * sizeof(uint32_t)
        Handle<CUDABuffer> vol_vert_num;
        Handle<CUDABuffer> vol_mc_scanned;// max_voxel_num * sizeof(uint32_t)
        //一般体数据生成的三角形数量相对于体素数量是很少的
        //dev
        Handle<CUDABuffer> vol_vert_pos;// max_voxel_num / 8 * sizeof(float3)

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

        _->ctx = info.gpu_mem_mgr->_get_cuda_context();
        //使用null流，如果和渲染同一个ctx，那么两者不能并行，即进行mc的时候不能渲染
        _->compute_stream = cub::cu_stream::null(info.gpu_mem_mgr->_get_cuda_context());

        _->max_voxel_num = info.max_voxel_num;
        _->max_vert_num = info.max_voxel_num / 8;

        _->vol_mc_code = info.gpu_mem_mgr->AllocBuffer(RescAccess::Unique, info.max_voxel_num * sizeof(uint32_t));
        _->vol_mc_scanned = info.gpu_mem_mgr->AllocBuffer(RescAccess::Unique, info.max_voxel_num * sizeof(uint32_t));
        _->vol_vert_num = info.gpu_mem_mgr->AllocBuffer(RescAccess::Unique, info.max_voxel_num * sizeof(uint32_t));

        _->vol_gen_host_vert = info.host_mem_mgr->AllocPinnedHostMem(RescAccess::Unique, info.max_voxel_num / 8 * sizeof(Float3), false);
        _->vol_vert_pos = info.gpu_mem_mgr->AllocBuffer(RescAccess::Unique, info.max_voxel_num / 8 * sizeof(Float3));

        _->params.vertex_pos = _->vol_vert_pos->view_1d<float3>(_->max_vert_num);
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
            cub::cu_kernel::pending(info, &MCKernel1_GenTriangles, params)
            .launch(_->compute_stream).check_error_on_throw();

            //将结果从dev传回host
            auto dev_ret = _->vol_vert_pos->view_1d<Float3>(total_vert_num);
            auto host_ret = _->vol_gen_host_vert->view_1d<Float3>(total_vert_num);
            cub::memory_transfer_info tf_info; tf_info.width_bytes = total_vert_num * sizeof(Float3);
            cub::cu_memory_transfer(dev_ret, host_ret, tf_info).launch(_->compute_stream).check_error_on_throw();

            mc_params.gen_host_vertices_ret = host_ret;

            return total_vert_num / 3;
        }
        catch (const std::exception& err) {
            LOG_ERROR("{}", err.what());
            return 0;
        }
    }

    void MarchingCubeAlgo::BindVTexture(VTextureHandle handle, TextureUnit unit) {
        assert(unit >= 0 && unit < MaxCUDATextureCountPerGPU);
//        if(_->params.cu_vtex[unit] != 0){
//            cudaDestroyTextureObject(_->params.cu_vtex[unit]);
//        }
        _->params.cu_vtex[unit] = handle->view_as({
            cub::e_clamp, cub::e_nearest, cub::e_normalized_float, false
        });

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


VISER_END
