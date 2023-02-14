#pragma once

#include "context.hpp"

CUB_BEGIN

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

    cu_context_handle_t create_context(uint32_t flags) const {
        return vutil::make_handle<cu_context>(*this, flags);
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
    CUdevice device{0};
    int device_id{-1};
};

inline cu_context::cu_context(cu_physical_device device, uint32_t flags) {
    CUB_CHECK(cuCtxCreate(&ctx, flags, device.device));
    set_ctx();
    this->device_id = device.get_device_id();
}

CUB_END