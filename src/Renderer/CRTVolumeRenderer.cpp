#undef UTIL_ENABLE_OPENGL

#include <Core/Renderer.hpp>

VISER_BEGIN

    namespace {
        using HashTableItem = GPUPageTableMgr::PageTableItem;
        static constexpr int HashTableSize = 1024;

        struct HashTable{
            HashTableItem hash_table[HashTableSize];
        };


        //...
        struct CUDAVolume{

        };

        struct CUDARenderParams{

        };
        //64kb


        struct CUDAPerFrameParams{

        };


        struct CUDAHashTable{
            uint4 hash_table[HashTableSize][2];
        };
        struct CUDAPageTable{
            CUDAHashTable* dev_hash_table;
        };
        struct CTRVolumeRenderKernelParams{
            CUDAVolume cu_volume;
            CUDARenderParams cu_render_params;
            CUDAPerFrameParams cu_per_frame_params;
            CUDAPageTable cu_page_table;
            CUDATexture cu_vtex[MaxCUDATextureCountPerGPU];

            FrameBuffer framebuffer;
        };
#ifdef UNIQUE_RENDER_PROCESS
        //todo
        __constant__ CUDAVolume cu_volume;
        __constant__ CUDARenderParams cu_render_params;
        __constant__ CUDAPerFrameParams cu_per_frame_params;
        __constant__ CUDAHashTable cu_hash_table;
        __constant__ CUDATexture cu_vtex[MaxCUDATextureCountPerGPU];
#endif

        CUB_GPU float4 RayCastVolume(const CTRVolumeRenderKernelParams& params){
            __shared__ uint4 hash_table[HashTableSize][2];
            // fill shared hash table

            __syncthreads();



        }

        CUB_KERNEL void CRTVolumeRenderKernel(CTRVolumeRenderKernelParams params){


        }
    }

    class CRTVolumeRendererPrivate{
    public:
        cub::cu_kernel kernel;


        CTRVolumeRenderKernelParams kernel_params;


        FrameBuffer Render(bool exclusive){
            cub::cu_kernel_launch_info launch_info;
            auto render_task = kernel.pending(launch_info, CRTVolumeRenderKernel, kernel_params);

        }
    };

    CRTVolumeRenderer::CRTVolumeRenderer(const CRTVolumeRenderer::CRTVolumeRendererCreateInfo &info) {

    }

    CRTVolumeRenderer::~CRTVolumeRenderer(){

    }

    void CRTVolumeRenderer::SetVolume(const VolumeInfo& volume_info) {

    }

    void CRTVolumeRenderer::SetRenderParams(const RenderParams& render_params) {

    }

    void CRTVolumeRenderer::SetPerFrameParams(const PerFrameParams &) {

    }

    FrameBuffer CRTVolumeRenderer::GetRenderFrame(bool exclusive) {
        return {};
    }

    void CRTVolumeRenderer::BindVTexture(VTextureHandle handle, TextureUnit unit) {

    }

    void CRTVolumeRenderer::BindPTBuffer(PTBufferHandle handle) {

    }

    void CRTVolumeRenderer::Lock() {

    }

    void CRTVolumeRenderer::UnLock() {

    }

    UnifiedRescUID CRTVolumeRenderer::GetUID() const {
        return 0;
    }


VISER_END