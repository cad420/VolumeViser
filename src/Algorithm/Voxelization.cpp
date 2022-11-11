#undef UTIL_ENABLE_OPENGL

#include <Algorithm/Voxelization.hpp>
#include "../Common/helper_math.h"
VISER_BEGIN

    namespace{
        static constexpr int HashTableSize = 1024;
        static constexpr int ThreadsPerBlocks = 64;
        static constexpr int MaxLodLevels = LevelOfDist::MaxLevelCount;
        using HashTableItem = GPUPageTableMgr::PageTableItem;

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
        struct SWCVoxelizeParams{
            cudaSurfaceObject_t cu_vsurf[MaxCUDATextureCountPerGPU];
            CUDAPageTable cu_page_table;
        };
        CUB_GPU void VirtualStore(const SWCVoxelizeParams& params,
                                  uint3 voxel_coord,
                                  uint32_t lod){

        }
        CUB_KERNEL void SWCVoxelizeKernel(SWCVoxelizeParams params){

        }
    }


    class SWCVoxelizerPrivate{
    public:



        std::mutex mtx;

        UnifiedRescUID uid;

        static UnifiedRescUID GenRescUID(){
            std::atomic<size_t> g_uid = 1;
            auto uid = g_uid.fetch_add(1);
            return uid;
        }
    };

    SWCVoxelizer::SWCVoxelizer(const VoxelizerCreateInfo& info) {
        _ = std::make_unique<SWCVoxelizerPrivate>();

        _->uid = _->GenRescUID();
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

    }

    void SWCVoxelizer::BindVTexture(VTextureHandle handle, TextureUnit unit) {
        handle->as_surface();
    }

    void SWCVoxelizer::BindPTBuffer(PTBufferHandle handle) {

    }


VISER_END


