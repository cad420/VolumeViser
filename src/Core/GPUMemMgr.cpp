#include <Core/GPUMemMgr.hpp>
#include <unordered_set>

VISER_BEGIN

    using GPUMemRescUID = GPUMemMgr::GPUMemRescUID;
    template<typename T>
    using Handle = GPUMemMgr::Handle<T>;
    namespace{
        template<typename T>
        struct TSharedResc{
            inline static std::unordered_map<GPUMemRescUID, Handle<T>> mp;
        };
    }

    class GPUMemMgrPrivate{
    public:
        size_t used_mem_bytes = 0;
        size_t max_mem_bytes = 0;

        int gpu_index = -1;
        cub::cu_context ctx;

        std::mutex mgr_mtx;

        struct SharedResc{
            std::unordered_map<GPUMemRescUID, Handle<CUDABuffer>> shared_buffer;
            std::unordered_map<GPUMemRescUID, Handle<CUDAPitchedBuffer>> shared_pitched_buffer;
        };
        //记录属于该实例的共享GPU模板化资源
        std::unordered_set<GPUMemRescUID> shared_resd_uid;

        static GPUMemRescUID GenUID(){
            static std::atomic<GPUMemRescUID> g_uid = 0;
            auto uid = g_uid.fetch_add(1);
            return uid;
        }
    };

    GPUMemMgr::GPUMemMgr(const GPUMemMgrCreateInfo& info) {
        _ = std::make_unique<GPUMemMgrPrivate>();

        auto devs = cub::cu_physical_device::get_all_device();
        bool find = false;
        auto init = [&](const cub::cu_physical_device& dev){
            _->ctx = dev.create_context(0);
            _->gpu_index = info.GPUIndex;
            _->max_mem_bytes = info.MaxGPUMemBytes;
        };
        for(auto& dev : devs){
            if(dev.get_device_id() == info.GPUIndex){
                find = true;
                init(dev);
                break;
            }
        }
        if(!find){
            throw ViserResourceCreateError("Create GPUMemMgr with invalid GPUIndex : " + std::to_string(info.GPUIndex));
        }

    }

    GPUMemMgr::~GPUMemMgr() {

    }

    void GPUMemMgr::Lock() {
        _->mgr_mtx.lock();
    }

    void GPUMemMgr::UnLock() {
        _->mgr_mtx.unlock();
    }

    template<typename T>
    GPUMemMgr::Handle<CUDAVolumeImage<T>> GPUMemMgr::AllocVolumeImage(GPUMemMgr::RescAccess access) {
        if(access == Unique){

        }
        else if(access == Shared){

        }
        else
            assert(false);
    }

    template<typename T>
    std::vector<GPUMemMgr::Handle<CUDAVolumeImage<T>>> GPUMemMgr::GetAllSharedVolumeImage() {
        return std::vector<Handle<CUDAVolumeImage<T>>>();
    }

    template<typename T, int N>
    GPUMemMgr::Handle<CUDAImage<T, N>> GPUMemMgr::AllocImage(GPUMemMgr::RescAccess access) {
        return GPUMemMgr::Handle<CUDAImage<T, N>>(GPUMemMgr::Unique, 0);
    }

    GPUMemMgr::Handle<CUDABuffer> GPUMemMgr::AllocBuffer(GPUMemMgr::RescAccess access) {
        return GPUMemMgr::Handle<CUDABuffer>(GPUMemMgr::Unique, 0);
    }

    GPUMemMgr::Handle<CUDAPitchedBuffer> GPUMemMgr::AllocPitchedBuffer(GPUMemMgr::RescAccess access) {
        return GPUMemMgr::Handle<CUDAPitchedBuffer>(GPUMemMgr::Unique, 0);
    }


    template<> GPUMemMgr::Handle<CUDAVolumeImage<uint8_t>> GPUMemMgr::AllocVolumeImage(GPUMemMgr::RescAccess access);
    template<> GPUMemMgr::Handle<CUDAVolumeImage<uint16_t>> GPUMemMgr::AllocVolumeImage(GPUMemMgr::RescAccess access);
VISER_END


