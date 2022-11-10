#undef UTIL_ENABLE_OPENGL

#include <Algorithm/MarchingCube.hpp>
#include "LookUpTable.cuh"
#include "../Common/helper_math.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

VISER_BEGIN

    namespace{

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


        static constexpr int HashTableSize = 1024;
        static constexpr int ThreadsPerBlocks = 64;
        using HashTableItem = GPUPageTableMgr::PageTableItem;
        struct CUDAPageTable{
            CUDABufferView1D<HashTableItem> table;
        };
        struct CUDAVolumeParams{
            float3 space;
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
            // 43 种情况对应的edge table起始地址
            CUDABufferView1D<const char*> tri_index_table;
            CUDABufferView1D<float4> vertex_pos;
            CUDABufferView1D<uint32_t> num_verts_scanned;
        };

        CUB_GPU float VirtualSampling(const MCKernelParams& params,
                              uint4 hash_table[][2],
                              uint3 voxel_coord, uint32_t lod){

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


#define MAKE_CODE0(caseIdx, subCaseIdx, triNum, configCaseIdx, subconfig13Value) (((((caseIdx << 4) | subCaseIdx) << 8) | triNum) | (configCaseIdx << 16) | (subconfig13Value << 24))
#define MAKE_CODE(caseIdx, subCaseIdx, triNum, configCaseIdx)  (((((caseIdx << 4) | subCaseIdx) << 8) | triNum) | (configCaseIdx << 16))
#define MAKE_HARD_CODE()

        //(可选)0.统计每个voxel是否会生成三角形
        //1.生成每个cube对应三角形的顶点，以及cube的case index
        CUB_KERNEL void MCKernel0_ClassifyVoxelAndGenVertices(MCKernelParams params){
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
            const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
             char * const p = nullptr;


            __shared__ const char*const* tri_edge_table[43];
            __shared__ const char*const*const* case13_edge_table[7];
            if(x == 0 && y == 0 && z == 0){
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
                case13_edge_table[1] = reinterpret_cast<const char*const*const*>(tiling13_3); // 13.3
                case13_edge_table[2] = reinterpret_cast<const char*const*const*>(tiling13_4); // 13.4
                case13_edge_table[3] = reinterpret_cast<const char*const*const*>(tiling13_5_1); // 13.5.1 or 13.5
                case13_edge_table[4] = reinterpret_cast<const char*const*const*>(tiling13_5_2); // 13.5.2 or 13.6
                case13_edge_table[5] = reinterpret_cast<const char*const*const*>(tiling13_3_); // 13.3 or 13.7
                case13_edge_table[6] = reinterpret_cast<const char*const*const*>(tiling13_2_); // 13.2 or 13.8
                tri_edge_table[41] = reinterpret_cast<const char*const*>(tiling13_1_); // 13.1 or 13.9
                tri_edge_table[42] = reinterpret_cast<const char*const*>(tiling14); // 14
            }
            __syncthreads();

            if(x >= params.cu_mc_params.shape.x
            || y >= params.cu_mc_params.shape.y
            || z >= params.cu_mc_params.shape.z)
                return;

            const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;

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

            uint3 voxel_coord = make_uint3(x, y, z) + params.cu_mc_params.origin;

            float field[8];
            field[0] = VirtualSampling(params, hash_table, voxel_coord, params.cu_mc_params.lod);
            field[1] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 0), params.cu_mc_params.lod);
            field[2] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 0), params.cu_mc_params.lod);
            field[3] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 0), params.cu_mc_params.lod);
            field[4] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 0, 1), params.cu_mc_params.lod);
            field[5] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 0, 1), params.cu_mc_params.lod);
            field[6] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(1, 1, 1), params.cu_mc_params.lod);
            field[7] = VirtualSampling(params, hash_table, voxel_coord + make_uint3(0, 1, 1), params.cu_mc_params.lod);

            uint32_t config_index = 0;
            config_index += uint32_t(field[0] < params.cu_mc_params.isovalue);
            config_index += uint32_t(field[1] < params.cu_mc_params.isovalue) << 1;
            config_index += uint32_t(field[2] < params.cu_mc_params.isovalue) << 2;
            config_index += uint32_t(field[3] < params.cu_mc_params.isovalue) << 3;
            config_index += uint32_t(field[4] < params.cu_mc_params.isovalue) << 4;
            config_index += uint32_t(field[5] < params.cu_mc_params.isovalue) << 5;
            config_index += uint32_t(field[6] < params.cu_mc_params.isovalue) << 6;
            config_index += uint32_t(field[7] < params.cu_mc_params.isovalue) << 7;

            const unsigned char case_idx = cases[config_index][0];
            const unsigned int config_idx_in_case = cases[config_index][1];
            unsigned int subconfig = 0, subconfig13Value;

            switch (case_idx) {
                // 0 : 生成 0 个三角形  内部 0 中情况
                case 0: {
                    params.vol_code.at(x, y, z) = MAKE_CODE(1, 0, 0, config_idx_in_case);
                    break;
                }

                // 1 : 生成 1 个三角形  内部 1 种情况
                case 1 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(1, 0, 1, config_idx_in_case);
                    break;
                }

                // 2 : 生成 2 个三角形  内部 1 种情况
                case 2 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(2, 0, 2, config_idx_in_case);
                    break;
                }
                // 3 : 内部 2 种情况  1:2  2:4
                case 3 : {
                    if(TestFace(field, test3[config_idx_in_case]) < 0){
                        params.vol_code.at(x, y, z) = MAKE_CODE(3, 0, 2, config_idx_in_case);
                    }
                    else{
                        params.vol_code.at(x, y, z) = MAKE_CODE(3, 1, 4, config_idx_in_case);
                    }
                    break;
                }
                // 4 : 内部 2 种情况  1:2  2:6
                case 4 : {
                    if(TestInterior(field, case_idx, test4[config_idx_in_case], 1) < 0){
                        params.vol_code.at(x, y, z) = MAKE_CODE(4, 0, 2, config_idx_in_case);
                    }
                    else{
                        params.vol_code.at(x, y, z) = MAKE_CODE(4, 1, 6, config_idx_in_case);
                    }
                    break;
                }
                // 5 : 生成 3 个三角形  内部 1 种情况
                case 5 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(5, 0, 3, config_idx_in_case);
                    break;
                }
                // 6 : 内部 3 种情况  1:3  2:9  3:5
                case 6 : {
                    if(TestFace(field, test6[config_idx_in_case][0]) < 0){
                        if(TestInterior(field, case_idx, test6[config_idx_in_case][1], test6[config_idx_in_case][2]) < 0){
                            params.vol_code.at(x, y, z) = MAKE_CODE(6, 0, 3, config_idx_in_case);
                        }
                        else{
                            params.vol_code.at(x, y, z) = MAKE_CODE(6, 1, 9, config_idx_in_case);
                        }
                    }
                    else{
                        params.vol_code.at(x, y, z) = MAKE_CODE(6, 2, 5, config_idx_in_case);
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
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 0, 3, config_idx_in_case);
                            break;
                        }
                        case 1 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 1, 5, config_idx_in_case);
                            break;
                        }
                        case 2 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 2, 5, config_idx_in_case);
                            break;
                        }
                        case 3 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 3, 9, config_idx_in_case);
                            break;
                        }
                        case 4 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 4, 5, config_idx_in_case);
                            break;
                        }
                        case 5 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 5, 9, config_idx_in_case);
                            break;
                        }

                        case 6 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(7, 6, 9, config_idx_in_case);
                            break;
                        }
                        case 7 : {
                            if(TestInterior(field, case_idx, test7[config_idx_in_case][3], test7[config_idx_in_case][4]) > 0){
                                params.vol_code.at(x, y, z) = MAKE_CODE(7, 7, 5, config_idx_in_case);
                            }
                            else{
                                params.vol_code.at(x, y, z) = MAKE_CODE(7, 8, 9, config_idx_in_case);
                            }
                            break;
                        }
                    }
                    break;
                }
                // 8 : 生成 2 个三角形  内部 1 种情况
                case 8 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(8, 0, 2, config_idx_in_case);
                    break;
                }
                // 9 : 生成 4 个三角形  内部 1 种情况
                case 9 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(9, 0, 4, config_idx_in_case);
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
                                params.vol_code.at(x, y, z) = MAKE_CODE(10, 0, 4, config_idx_in_case);
                            }
                            else{
                                params.vol_code.at(x, y, z) = MAKE_CODE(10, 1, 8, config_idx_in_case);
                            }
                            break;
                        }

                        case 1 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(10, 2, 8, config_idx_in_case);
                            break;
                        }

                        case 2 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(10, 3, 8, config_idx_in_case);
                            break;
                        }

                        case 3 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(10, 4, 4, config_idx_in_case);

                            break;
                        }
                    }
                    break;
                }
                // 11 :生成 4 个三角形  内部 1 种情况
                case 11 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(11, 0, 4, config_idx_in_case);
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
                                params.vol_code.at(x, y, z) = MAKE_CODE(12, 0, 4, config_idx_in_case);
                            }
                            else{
                                params.vol_code.at(x, y, z) = MAKE_CODE(12, 1, 8, config_idx_in_case);
                            }
                            break;
                        }

                        case 1 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(12, 2, 8, config_idx_in_case);
                            break;
                        }

                        case 2 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(12, 3, 8, config_idx_in_case);
                            break;
                        }

                        case 3 : {
                            params.vol_code.at(x, y, z) = MAKE_CODE(12, 4, 4, config_idx_in_case);
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
                        params.vol_code.at(x, y, z) = MAKE_CODE(13, 0, 4, config_idx_in_case);
                    }
                    else if(subconfig13Value < 7){
                        params.vol_code.at(x, y, z) = MAKE_CODE0(13, 1, 6, config_idx_in_case, subconfig13Value);
                    }
                    else if(subconfig13Value < 19){
                        params.vol_code.at(x, y, z) = MAKE_CODE0(13, 2, 10, config_idx_in_case, subconfig13Value);
                    }
                    else if(subconfig13Value < 23){
                        params.vol_code.at(x, y, z) = MAKE_CODE0(13, 3, 12, config_idx_in_case, subconfig13Value);
                    }
                    else if(subconfig13Value < 27){
                        if(TestInterior(field, case_idx, test13[config_idx_in_case][6], tiling13_5_1[config_idx_in_case][subconfig13Value - 23][0]) < 0){
                            params.vol_code.at(x, y, z) = MAKE_CODE0(13, 4, 6, config_idx_in_case, subconfig13Value);
                        }
                        else{
                            params.vol_code.at(x, y, z) = MAKE_CODE0(13, 5, 10, config_idx_in_case, subconfig13Value);
                        }

                    }
                    else if(subconfig13Value < 39){
                        params.vol_code.at(x, y, z) = MAKE_CODE0(13, 6, 10, config_idx_in_case, subconfig13Value);
                    }
                    else if(subconfig13Value < 45){
                        params.vol_code.at(x, y, z) = MAKE_CODE0(13, 7, 6, config_idx_in_case, subconfig13Value);
                    }
                    else if(subconfig13Value == 45){
                        params.vol_code.at(x, y, z) = MAKE_CODE0(13, 8, 4, config_idx_in_case, subconfig13Value);
                    }
                    break;
                }
                // 14 : 生成 4 个三角形  内部 1 种情况
                case 14 : {
                    params.vol_code.at(x, y, z) = MAKE_CODE(14, 0, 4, config_idx_in_case);
                    break;
                }
            }

        }

        __forceinline__ CUB_GPU float3 VertexInterp(float isovalue, float3 p0, float3 p1, float f0, float f1){
            return lerp(p0, p1, (isovalue - f0) / (f1 - f0));
        }

        CUB_KERNEL void MCKernel1_GenTriangleIndices(MCKernelParams params){
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

            if(x >= params.cu_mc_params.shape.x
               || y >= params.cu_mc_params.shape.y
               || z >= params.cu_mc_params.shape.z)
                return;

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

            float3 p = make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z) * params.cu_vol_params.space;
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
            vert_list[0] = VertexInterp(params.cu_mc_params.isovalue, vert[0], vert[1], field[0], field[0]);
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

            const int case_idx = code >> 12;
            const int sub_case_dix = (code >> 8) & 0xf;
            const int m_case_idx = (code >> 8) &0xff;
            const int tri_num = code & 0xff;
            const int config_idx_in_case = (code >> 16) & 0xff;
            const int subconfig13_val = (code >> 24) & 0xff;
            const unsigned char* edge_table;
            switch(case_idx){
                case 0 : {
                    edge_table = nullptr;
                    break;
                }

                case 1 : {
                    edge_table = reinterpret_cast<const unsigned char *>(tiling1[config_idx_in_case]);
                    break;
                }

                case 2 : {
                    edge_table = reinterpret_cast<const unsigned char *>(tiling2[config_idx_in_case]);
                    break;
                }

                case 3 : {
                    switch (sub_case_dix) {
                        case 0 : {
                            edge_table = reinterpret_cast<const unsigned char *>(tiling3_1[config_idx_in_case]);
                            break;
                        }
                        case 1 : {
                            edge_table = reinterpret_cast<const unsigned char *>(tiling3_2[config_idx_in_case]);
                            break;
                        }
                    }
                }

            }

            for(int i = 0; i < tri_num; i++){
                int a = edge_table[i * 3];
                int b = edge_table[i * 3 + 1];
                int c = edge_table[i * 3 + 2];

                uint32_t index = params.num_verts_scanned.at(voxel_index) + i;

                //check index ?

                params.vertex_pos.at(index) = make_float4(vert_list[a], 1.f);
                params.vertex_pos.at(index + 1) = make_float4(vert_list[b], 1.f);
                params.vertex_pos.at(index + 2) = make_float4(vert_list[c], 1.f);
            }

        }

    }

    static UnifiedRescUID GenMCAlgoUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::MCAlgo);
    }


    class MarchingCubeAlgoPrivate{
    public:
        cub::cu_context ctx;

        cub::cu_stream compute_stream;

        MCKernelParams params;

        Handle<CUDABuffer> vol_mc_code;
        Handle<CUDABuffer> vol_mc_scanned;

        UnifiedRescUID uid;

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

        // init gpu edge table
        {

        }

        _->uid = GenMCAlgoUID();
    }


    MarchingCubeAlgo::~MarchingCubeAlgo() {

    }

    struct BOP{
        CUB_GPU uint32_t operator()(uint32_t a, uint32_t b) const {
            return a + (b & 0xff);
        }
    };
    void MarchingCubeAlgo::Run(const MarchingCubeAlgoParams &params) {
        //一个线程复杂一个cube，一个block共用page table
        const dim3 tile = {4u, 4u, 4u};
        const dim3 grid_dim = {
                (params.shape.x + tile.x - 1) / tile.x,
                (params.shape.y + tile.y - 1) / tile.y,
                (params.shape.z + tile.z - 1) / tile.z
        };
        cub::cu_kernel_launch_info info{grid_dim, tile};

        size_t num_voxels;

        try{
            void* params[] = {&_->params};
            cub::cu_kernel::pending(info, &MCKernel0_ClassifyVoxelAndGenVertices, params)
            .launch(_->compute_stream).check_error_on_throw();

            auto vol_mc_code_view = _->vol_mc_code->view_1d<uint32_t>(num_voxels);

            auto vol_mc_scanned_view = _->vol_mc_scanned->view_1d<uint32_t>(num_voxels);

            thrust::exclusive_scan(thrust::device_ptr<uint32_t>(vol_mc_code_view.data()), thrust::device_ptr<uint32_t>(vol_mc_code_view.data() + num_voxels),
                    thrust::device_ptr<uint32_t>(vol_mc_scanned_view.data()), 0, BOP{});

            cub::cu_kernel::pending(info, &MCKernel1_GenTriangleIndices, params)
            .launch(_->compute_stream).check_error_on_throw();
        }
        catch (const std::exception& err) {
            LOG_ERROR("{}", err.what());
        }
    }

    void MarchingCubeAlgo::BindVTexture(VTextureHandle handle, TextureUnit unit) {

    }

    void MarchingCubeAlgo::BindPTBuffer(PTBufferHandle handle) {

    }


VISER_END
