#pragma once

#define VISER_BEGIN namespace viser{
#define VISER_END }

// cuda
#include <cuda_wrapper.hpp>

// util
#include <util.hpp>

// stl
#include <vector>
#include <memory>

VISER_BEGIN

template<typename T, int N>
using CUDAArray = cub::cu_array<T, N>;

template<typename T, int N>
using CUDAImage = CUDAArray<T, N>;

template<typename T>
using CUDAVolumeImage = CUDAImage<T, 3>;

using CUDABuffer = cub::cu_buffer<false>;

using CUDAHostBuffer = CUDABuffer;

using CUDAPitchedBuffer = cub::cu_buffer<true>;

template<typename T>
using CUDABufferView1D = cub::buffer_view<T, 1>;

template<typename T>
using CUDABufferView2D = cub::buffer_view<T, 2>;

template<typename T>
using CUDABufferView3D = cub::buffer_view<T, 3>;

using CUDATexture = cub::cu_texture;


using Float3 = vutil::vec3f;

using Int3 = vutil::tvec3<int>;
using UInt3 = vutil::tvec3<uint32_t>;

using Mat3 = vutil::tmat3_c<float>;
using Mat4 = vutil::tmat4_c<float>;

using Transform = vutil::mat4f_c::right_transform;

using BoundingBox3D = vutil::aabb3f;
using Frustum = vutil::frustum_extf;
using vutil::BoxVisibility;

// Viser
class GPUMemMgr;
class HostMemMgr;
class ResourceMgr;
class FixedHostMemMgr;
class GPUPageTableMgr;
class GPUVTexMgr;
class Renderer;
class DistributeMgr;


class ViserFileOpenError : public std::exception{
public:
    ViserFileOpenError(const std::string& msg) : std::exception(msg.c_str()){}
};

class ViserResourceCreateError : public std::exception{
public:
    ViserResourceCreateError(const std::string& msg) : std::exception(msg.c_str()){}
};

    enum class RescAccess{
        Unique,
        Shared
    };

    using UnifiedRescUID = size_t;
    inline UnifiedRescUID INVALID_RESC_ID = 0ull;

    enum class UnifiedRescType : uint32_t{
        Unknown = 0u,
        HostMemMgr = 1 ,
        GPUMemMgr = 2,
        FixedHostMemMgr = 3,
        CUDAMem = 4,
        HostMem = 5,
        MaxEnum = 6
    };

    //只检查uid的type是否是UnifiedRescType中的，以及是否等于INVALID_RESC_ID
    inline bool CheckUnifiedRescUID(UnifiedRescUID uid){
        if(uid == INVALID_RESC_ID) return false;
        uint32_t type = uid >> 32;
        if(static_cast<UnifiedRescType>(type) == UnifiedRescType::Unknown ||
           type >= static_cast<uint32_t>(UnifiedRescType::MaxEnum))
            return false;

        return true;
    }

    class ThreadSafeBase{
    public:
        virtual void Lock() = 0;

        virtual void UnLock() = 0;
    };

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

        T& operator*(){
            return *obj;
        }

        const T& operator*() const{
            return *obj;
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

    //当释放资源的时候，应该让资源分配池知道
    //对于Unique资源，只能移动拷贝不能赋值拷贝
    //对于Shared资源，需要在所有资源都释放完后通知资源池
    // 如果是Shared的资源，提供加锁操作，Unique资源总是能加锁成功，而Shared则不一定
    template<typename T>
    class Handle{
    public:
        struct AccessLocker{
            AccessLocker(std::unique_lock<std::mutex> lk)
                    :lk(std::move(lk))
            {}
            ~AccessLocker(){
                UnLock();
            }
            bool IsLocked() const{
                return lk.owns_lock();
            }
            void UnLock(){
                if(lk.owns_lock())
                    lk.unlock();
            }
        private:
            std::unique_lock<std::mutex> lk;
        };

        AccessLocker Lock(bool wait){
            if(_->access == RescAccess::Unique){
                // always success
                return {std::unique_lock<std::mutex>(_->mtx)};
            }
            else if(_->access == RescAccess::Shared){
                if(wait){
                    std::unique_lock<std::mutex> lk(_->mtx);
                    return {std::move(lk)};
                }
                else{
                    std::unique_lock<std::mutex> lk(_->mtx, std::try_to_lock_t());
                    return {std::move(lk)};
                }
            }
            else{
                assert(false);
            }
        }

        size_t UID() const{
            return _->uid;
        }

        RescAccess AccessType() const{
            return _->access;
        }

        T* operator->(){
            return _->resc.get();
        }

        const T* operator->() const{
            return _->resc.get();
        }

        Handle(RescAccess access, size_t uid, std::shared_ptr<T> resc)
                :_(std::make_shared<Inner>())
        {
            _->access = access;
            _->uid = uid;
            _->resc = std::move(resc);
        }

        Handle() = default;

        ~Handle(){
            Destroy();
        }

        Handle(const Handle& other){
            if(_->access == RescAccess::Unique){
                assert(false);
            }
            else if(_->access == RescAccess::Shared){
                _ = other._;
            }
            else
                assert(false);
        }

        Handle& operator=(const Handle& other){
            if(_->access == RescAccess::Unique){
                assert(false);
            }
            else if(_->access == RescAccess::Shared){
                Destroy();
                new(this) Handle(other);
                return *this;
            }
            else
                assert(false);
            return *this;
        }

        Handle(Handle&& other) noexcept{
            _ = std::move(other._);
        }

        Handle& operator=(Handle&& other) noexcept{
            new(this) Handle(std::move(other));
            return *this;
        }


        //统一销毁资源，不管是Unique还是Shared的
        //应该在没有被锁住的情况下调用
        void Destroy(){
            _->callback(_->uid);
            _->resc.reset();
            _.reset();
        }

        bool IsValid() const{
            return _ && _->resc && CheckUnifiedRescUID(_->uid);
        }
    private:
        struct Inner{
            std::mutex mtx;
            RescAccess access;
            std::shared_ptr<T> resc;
            UnifiedRescUID uid = 0;
            std::function<void(UnifiedRescUID)> callback;
        };
    private:
        std::shared_ptr<Inner> _;
    };


    inline void ExtractFrustumFromMatrix(const Mat4& matrix, Frustum& frustum){
        vutil::extract_frustum_from_matrix(matrix, frustum, true);
    }

    inline BoxVisibility GetBoxVisibility(const Frustum& frustum, const BoundingBox3D& box){
        return vutil::get_box_visibility(frustum, box);
    }

VISER_END