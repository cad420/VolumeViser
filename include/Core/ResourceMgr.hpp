#pragma once


#include <Core/GPUMemMgr.hpp>
#include <Core/HostMemMgr.hpp>

VISER_BEGIN

//除了包装引用之外，增加一些自动Lock、UnLock的操作
template<typename T>
class Ref{
public:
    Ref() = default;

    Ref(T* p)
    :obj(p)
    {
        assert(p);
        obj->Lock();
    }

    ~Ref(){
        Release();
    }

    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;

    Ref(Ref&& other)noexcept{
        obj = other.obj;
        other.obj = nullptr;
    }
    Ref& operator=(Ref&& other) noexcept{
        Release();
        new(this) Ref(std::move(other));
        return *this;
    }

    T* operator->(){
        assert(obj);
        return obj;
    }

    const T* operator->() const {
        assert(obj);
        return obj;
    }
    //手动释放，之后不能再访问，否则触发assert
    void Release(){
        if(obj){
            obj->UnLock();
            obj = nullptr;
        }
    }
private:
    T* obj = nullptr;
};
class ResourceMgrPrivate;
class ResourceMgr final{
public:
    enum ResourceType{
        Host,
        Device
    };
    struct ResourceDesc{
        ResourceType type;
        size_t MaxMemBytes = 0;
        int DeviceIndex;
    };

    using UID = size_t;

    //失败会抛出异常
    UID RegisterResourceMgr(ResourceDesc desc);

    std::vector<UID> GetAll() const;

    bool Exist(UID uid) const;

    bool Exist(UID uid, ResourceType type) const;

    template<typename T, ResourceType type>
    Ref<T> GetResourceMgrRef(UID) = delete;

    template<>
    Ref<GPUMemMgr> GetResourceMgrRef<GPUMemMgr, Device>(UID);

    template<>
    Ref<HostMemMgr> GetResourceMgrRef<HostMemMgr, Host>(UID);



    static ResourceMgr& GetInstance();

    ~ResourceMgr();

private:
    ResourceMgr();

    std::unique_ptr<ResourceMgrPrivate> _;
};


VISER_END
