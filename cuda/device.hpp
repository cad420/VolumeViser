#pragma once

#include "context.hpp"

CUB_BEGIN

namespace detail {
    struct cu_init {
        cu_init(){
            static auto _ = []{
                try{
                    CUB_CHECK(cuInit(0));
                    CUB_WHEN_DEBUG(std::cout << "CUDA init successfully..." << std::endl)
                }
                catch (const std::exception& err) {
                    std::cerr << err.what() << std::endl;
                    exit(0);
                }
                return 0;
            }();
        }
    };
    inline cu_init _cu_init = cu_init();
}

class cu_physical_device{
public:
    friend class cu_context;

    explicit cu_physical_device(int device_id)
    :device_id(device_id)
    {
        CUB_CHECK(cuDeviceGet(&device, device_id));
    }

    static std::vector<cu_physical_device> get_all_device(){
        std::vector<cu_physical_device> devices;
        int device_count = 0;
        CUB_CHECK(cuDeviceGetCount(&device_count));
        for(int id = 0; id < device_count; ++id)
            devices.emplace_back(id);
        return devices;
    }

    cu_context create_context(uint32_t flags) const {
        return cu_context(*this, flags);
    }

    int get_device_id() const {
        return device_id;
    }

    std::string get_device_name() const {
        std::string name(64, ' ');
        CUB_CHECK(cuDeviceGetName(name.data(), name.length(), device));
        return name;
    }

private:
    CUdevice device;
    int device_id;
};

inline cu_context::cu_context(cu_physical_device device, uint32_t flags) {
    CUB_CHECK(cuCtxCreate(&_->ctx, flags, device.device));
    set_ctx();
}

CUB_END