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
using HostBuffer = std::vector<uint8_t>;

using CUDAPitchedBuffer = cub::cu_buffer<true>;

template<typename T>
using CUDABufferView1D = cub::buffer_view<T, 1>;

template<typename T>
using CUDABufferView2D = cub::buffer_view<T, 2>;

template<typename T>
using CUDABufferView3D = cub::buffer_view<T, 3>;

using CUDATex = cub::cu_texture;
using CUDATexture = cub::cu_texture_wrap;
constexpr int MaxCUDATextureCountPerGPU = 32;

template<typename T>
struct GeneralRescTraits{
    static constexpr bool value = false;
};

struct FrameBuffer;

template<> struct GeneralRescTraits<CUDABuffer>       { static constexpr bool value = true; };
template<> struct GeneralRescTraits<CUDAPitchedBuffer>{ static constexpr bool value = true; };
template<> struct GeneralRescTraits<CUDATexture>      { static constexpr bool value = true; };
template<> struct GeneralRescTraits<HostBuffer>       { static constexpr bool value = true; };
template<> struct GeneralRescTraits<FrameBuffer>      { static constexpr bool value = true; };

template<typename T>
inline constexpr bool IsGeneralResc = GeneralRescTraits<T>::value;

using Float3 = vutil::vec3f;
using Float4 = vutil::vec4f;

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
class HashPageTable;
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
inline constexpr UnifiedRescUID INVALID_RESC_ID = 0ull;

enum class UnifiedRescType : uint8_t{
    Unknown = 0,
    RescMgr = 1,
    HostMemMgr = 2 ,
    GPUMemMgr = 3,
    FixedHostMemMgr = 4,
    VolumeIO = 5,
    SWCIO = 6,
    MeshIO = 7,
    GridVolume = 8,
    GridVolumeBlock = 9,
    GPUPageTableMgr = 10,
    GPUVTexMgr = 11,
    CRTVolRenderer = 12,
    General = 13,
    MCAlgo = 14,
    MaxEnum = 15
};

//最高位8位代表资源类型，低56位代表一个全局唯一该类型资源的编号
inline UnifiedRescUID GenUnifiedRescUID(size_t uid, UnifiedRescType type){
    uint8_t t = static_cast<uint8_t >(uid >> 56);
    assert(t == 0 && type != UnifiedRescType::Unknown);
    t = static_cast<uint8_t>(type);
    assert(t < static_cast<uint8_t>(UnifiedRescType::MaxEnum));
    return uid | ((size_t)t) << 56;
}

//每种类型的资源编号默认0为非法编号
inline UnifiedRescUID GenInvalidUnifiedRescUID(UnifiedRescType type){
    return GenUnifiedRescUID(0, type);
}

inline UnifiedRescUID GenGeneralUnifiedRescUID(){
    static std::atomic<size_t> g_uid = 1;
    auto uid = g_uid.fetch_add(1);
    return GenUnifiedRescUID(uid, UnifiedRescType::General);
}

//检查uid的type是否是UnifiedRescType中的，是否等于INVALID_RESC_ID，以及资源编号是否为0
inline bool CheckUnifiedRescUID(UnifiedRescUID uid){
    if(uid == INVALID_RESC_ID) return false;
    uint8_t type = uid >> 56;
    if(static_cast<UnifiedRescType>(type) == UnifiedRescType::Unknown ||
       type >= static_cast<uint8_t>(UnifiedRescType::MaxEnum))
        return false;
    if((uid & ((1ull << 56) - 1)) == 0) return false;
    return true;
}

class UnifiedRescBase{
public:
    virtual void Lock() = 0;

    virtual void UnLock() = 0;

    virtual UnifiedRescUID GetUID() const = 0;
};

//除了包装引用之外，增加一些自动Lock、UnLock的操作
template<typename T>
class Ref : vutil::no_copy_t{
public:
    Ref() = default;

    Ref(T* p, bool lock = true)
    :obj(p), locked(lock)
    {
        if(lock)
            obj->Lock();
    }

    ~Ref(){
        Release();
    }

    Ref(Ref&& other)noexcept{
        obj = other.obj;
        locked = other.locked;
        other.obj = nullptr;
    }

    Ref& operator=(Ref&& other) noexcept{
        Release();
        new(this) Ref(std::move(other));
        return *this;
    }

    T* _get_ptr() {
        return obj;
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
        if(locked && obj){
            obj->UnLock();
            obj = nullptr;
            locked = false;
        }
    }

    bool IsLocked() const{
        return locked;
    }

    bool IsValid() const{
        return obj;
    }

private:
    bool locked = false;
    T* obj = nullptr;
};

// 当释放资源的时候，应该让资源分配池知道
// 对于Unique资源，只能移动拷贝不能赋值拷贝
// 对于Shared资源，需要在所有资源都释放完后通知资源池
// Unique资源总是能加锁成功，而Shared则不一定
enum class AccessType{
    Read,
    Write
};

template<typename T>
class Handle{
public:
    struct AccessLocker{
        AccessLocker(std::function<void()> f)
        :s(std::move(f))
        {}

        ~AccessLocker(){
            UnLock();
        }

        void UnLock(){
            s.call();
        }

    private:
        vutil::scope_bomb_t<std::function<void()>> s;
    };

    AccessLocker AccessLock(AccessType type){
        if(_->access == RescAccess::Unique){
            // always success
            return {nullptr};
        }
        else if(_->access == RescAccess::Shared){
            if(type == AccessType::Read){
                _->rw_lk.lock_read();
                return AccessLocker([&](){
                   _->rw_lk.unlock_read();
                });
            }
            else if(type == AccessType::Write){
                _->rw_lk.lock_write();
                return AccessLocker([&](){
                    _->rw_lk.unlock_write();
                });
            }
            else{
                assert(false);
                return {nullptr};
            }
        }
        else{
            assert(false);
            return {nullptr};
        }
    }


    UnifiedRescUID GetUID() const{
        return _->uid;
    }

    void SetUID(UnifiedRescUID uid){
//        assert(IsValid());
        _->uid = uid;
    }

    RescAccess GetRescAccess() const{
        return _->access;
    }

    T* operator->(){
        return _->resc.get();
    }

    const T* operator->() const{
        return _->resc.get();
    }

    T& operator*(){
        return *_->resc;
    }

    const T& operator*() const {
        return *_->resc;
    }

    Handle(UnifiedRescUID uid, RescAccess access, std::shared_ptr<T> resc)
            :_(std::make_shared<Inner>())
    {
        _->access = access;
        _->uid = uid;
        _->resc = std::move(resc);
    }

    Handle(RescAccess access, std::shared_ptr<T> resc)
    :_(std::make_shared<Inner>())
    {
        if constexpr(IsGeneralResc<T>){
            _->access = access;
            _->uid = GenGeneralUnifiedRescUID();
            _->resc = std::move(resc);
        }
        else{
            static_assert(std::is_base_of_v<UnifiedRescBase, T>, "");
            _->access = access;
            _->uid = resc->GetUID();
            _->resc = std::move(resc);
        }

    }

    Handle() = default;

    ~Handle(){

    }

    Handle(const Handle& other){
        if(_){
            if (_->access == RescAccess::Unique) {
                assert(false);
            } else if (_->access == RescAccess::Shared) {
                _ = other._;
            } else
                assert(false);
        }
        else{
            _ = other._;
        }
    }

    Handle& operator=(const Handle& other){
        if(_){
            if (_->access == RescAccess::Unique) {
                assert(false);
            } else if (_->access == RescAccess::Shared) {
                Destroy();
                new(this) Handle(other);
                return *this;
            } else
                assert(false);
        }
        else{
            Destroy();
            _ = other._;
        }
        return *this;
    }

    Handle(Handle&& other) noexcept{
        _ = std::move(other._);
    }

    Handle& operator=(Handle&& other) noexcept{
        new(this) Handle(std::move(other));
        return *this;
    }

    Handle& SetCallback(std::function<void(UnifiedRescUID)> callback){
        _->callback = std::move(callback);
        return *this;
    }

    //统一销毁资源，不管是Unique还是Shared的
    //应该在没有被锁住的情况下调用
    void Destroy(){
        if(_){
            if(_->callback)
                _->callback(_->uid);
            _->resc.reset();
            _.reset();
        }
    }

    bool IsValid() const{
        return _ && _->resc && CheckUnifiedRescUID(_->uid);
    }

    bool IsReadLocked(){
        return _->rw_lk.is_read_locked();
    }
    bool IsWriteLocked(){
        return _->rw_lk.is_write_locked();
    }

    void AddReadLock(){
        _->rw_lk.lock_read();
    }

    void AddWriteLock(){
        _->rw_lk.lock_write();
    }

    void ReleaseReadLock(){
        _->rw_lk.unlock_read();
    }

    void ReleaseWriteLock(){
        _->rw_lk.unlock_write();
    }

    void ConvertWriteToReadLock(){
        _->rw_lk.converse_write_to_read();
    }

private:
    struct Inner{
        vutil::rw_spinlock_t rw_lk;
        RescAccess access = RescAccess::Shared;
        std::shared_ptr<T> resc;
        UnifiedRescUID uid = INVALID_RESC_ID;
        std::function<void(UnifiedRescUID)> callback;
    };

    std::shared_ptr<Inner> _;
};

template<typename T, typename... Args>
auto NewHandle(RescAccess access, Args&&... args){
    return Handle<T>(access, std::make_shared<T>(std::forward<Args>(args)...));
}

template<typename T, typename... Args>
auto NewHandle(UnifiedRescUID uid, RescAccess access, Args&&... args){
    return Handle<T>(uid, access, std::make_shared<T>(std::forward<Args>(args)...));
}

template<typename T, typename... Args>
auto NewGeneralHandle(RescAccess access, Args&&... args){
    return Handle<T>(access, std::make_shared<T>(std::forward<Args>(args)...));
}

inline void ExtractFrustumFromMatrix(const Mat4& matrix, Frustum& frustum){
    vutil::extract_frustum_from_matrix(matrix, frustum, true);
}

inline BoxVisibility GetBoxVisibility(const Frustum& frustum, const BoundingBox3D& box){
    return vutil::get_box_visibility(frustum, box);
}

inline BoundingBox3D FrustumToBoundingBox3D(const Frustum& frustum){
    BoundingBox3D box;
    for(int i = 0; i < 8; i++){
        box.low.x = std::min(box.low.x, frustum.frustum_corners[i].x);
        box.low.y = std::min(box.low.y, frustum.frustum_corners[i].y);
        box.low.z = std::min(box.low.z, frustum.frustum_corners[i].z);
        box.high.x = std::max(box.high.x, frustum.frustum_corners[i].x);
        box.high.y = std::max(box.high.y, frustum.frustum_corners[i].y);
        box.high.z = std::max(box.high.z, frustum.frustum_corners[i].z);
    }
    return box;
}

VISER_END