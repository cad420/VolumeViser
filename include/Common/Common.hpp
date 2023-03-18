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

using CUDAContext = cub::cu_context_handle_t;

using CUDAStream = cub::cu_stream;

using CUDAKernel = cub::cu_kernel;

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

using CUDASurface = cub::cu_surface;

template<typename T>
struct GeneralRescTraits{
    static constexpr bool value = false;
};

struct FrameBuffer;

template<> struct GeneralRescTraits<CUDABuffer>       { static constexpr bool value = true; };
template<> struct GeneralRescTraits<CUDAPitchedBuffer>{ static constexpr bool value = true; };
template<> struct GeneralRescTraits<CUDATexture>      { static constexpr bool value = true; };
template<> struct GeneralRescTraits<CUDASurface>      { static constexpr bool value = true; };
template<> struct GeneralRescTraits<HostBuffer>       { static constexpr bool value = true; };
template<> struct GeneralRescTraits<FrameBuffer>      { static constexpr bool value = true; };

template<typename T>
inline constexpr bool IsGeneralResc = GeneralRescTraits<T>::value;

using Float2 = vutil::vec2f;
using Float3 = vutil::vec3f;
using Float4 = vutil::vec4f;

using Int2 = vutil::tvec2<int>;
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

//todo
#define G_HashTableSize  1024;

class ViserFileOpenError : public std::exception{
public:
    ViserFileOpenError(const std::string& msg) : std::exception(msg.c_str()){}
};

class ViserResourceCreateError : public std::exception{
public:
    ViserResourceCreateError(const std::string& msg) : std::exception(msg.c_str()){}
};

enum class ResourceType
{
    Object,
    Buffer
};

using UnifiedRescUID = size_t;
inline constexpr UnifiedRescUID INVALID_RESC_ID = 0ull;

enum class UnifiedRescType : uint8_t{
    Unknown = 0,
    RescMgr,
    HostMemMgr,
    GPUMemMgr,
    FixedHostMemMgr,
    GPUPageTableMgr,
    GPUVTexMgr,
    DistributeMgr,
    VolumeIO,
    SWCIO,
    MeshIO,
    GridVolume,
    GridVolumeBlock,
    GridVoxelizeBlock,
    CRTVolRenderer,
    RTVolRenderer,
    PBVolRenderer,
    MCAlgo,
    SWCVoxelizeAlgo,
    SWC,
    Mesh,
    General,
    MaxEnum = 255
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

inline bool CheckGeneralUnifiedResc(UnifiedRescUID uid) {
    if(uid == INVALID_RESC_ID) return false;
    uint8_t type = uid >> 56;
    return static_cast<UnifiedRescType>(type) == UnifiedRescType::General;
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

template<typename T>
class Ref {
public:


    Ref() = default;

    Ref(T* p, bool safe = false)
    :obj(p), thread_safe(safe)
    {
        assert(p);
    }

    Ref(const Ref<T>& other) = default;
    Ref& operator=(const Ref<T>& other) = default;

    ~Ref(){
        Release();
    }

    T* _get_ptr() {
        return obj;
    }


    T& operator*(){
        assert(obj);
        return *obj;
    }

    const T& operator*() const {
        assert(obj);
        return *obj;
    }

    template<typename F, typename... Args>
    decltype(auto) Invoke(F&& f, Args&&... args){
        auto _ = AutoLock();
        return (obj->*f)(std::forward<Args>(args)...);
    }


    Ref<T>& LockRef() {
        thread_safe = true;
        return *this;
    }

    void Release(){
        if(obj)
            obj = nullptr;
    }

    bool IsThreadSafe() const {
        return thread_safe;
    }

    bool IsValid() const {
        return obj;
    }
    auto AutoLock() const {
        if constexpr(std::is_base_of_v<UnifiedRescBase, T>){
            if(thread_safe){
                obj->Lock();
            }
            return vutil::scope_bomb_t([this]{
                if(thread_safe) obj->UnLock();
            });
        }
        else{
            return ;//vutil::scope_bomb_t([this]{});
        }
    }
  private:
    bool thread_safe = false;
    T* obj = nullptr;
};

enum class AccessType{
    Read,
    Write
};


template<typename T>
class Handle{
    struct Inner{
        using ReadWriteLock = vutil::rw_spinlock_t;
        ReadWriteLock rw_lk;
        ResourceType type;
        std::shared_ptr<T> resc;
        UnifiedRescUID uid = INVALID_RESC_ID;
    };
    std::shared_ptr<Inner> _;
public:
    Handle() = default;

    Handle(ResourceType type, std::shared_ptr<T> resc)
    :_(std::make_shared<Inner>())
    {
        _->resc = std::move(resc);
        _->type = type;
        if(type == ResourceType::Buffer){
            _->uid = GenGeneralUnifiedRescUID();
        }
    }

    ~Handle() = default;

    Handle(const Handle& other) = default;

    Handle& operator=(const Handle& other){
        Destroy();
        new(this) Handle(other);
        return *this;
    }

    Handle(Handle&& other) noexcept{
        _ = std::move(other._);
    }

    Handle& operator=(Handle&& other) noexcept{
        Destroy();
        new(this) Handle(std::move(other));
        return *this;
    }

    //加互斥锁
    auto AutoLocker() const {
        if constexpr(std::is_base_of_v<UnifiedRescBase,T>){
            _->resc->Lock();
            return vutil::scope_bomb_t([this]{
                _->resc->UnLock();
            });
        }
        else{
            _->rw_lk.lock_write();
            return vutil::scope_bomb_t([this]{
                _->rw_lk.unlock_write();
            });
        }
    }

    UnifiedRescUID GetUID() const{
        return _->uid;
    }

    Handle<T>& SetUID(UnifiedRescUID uid) {
        _->uid = uid;
        return *this;
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

    void Destroy(){
        _.reset();
    }

    bool IsValid() const{
        return _ && _->resc.get();
    }

    bool IsLocked(){
        return IsReadLocked() || IsWriteLocked();
    }

    bool IsReadLocked(){
        return _->rw_lk.is_read_locked();
    }
    bool IsWriteLocked(){
        return _->rw_lk.is_write_locked();
    }

    Handle<T>& AddReadLock(){
        _->rw_lk.lock_read();
        return *this;
    }

    Handle<T>& AddWriteLock(){
        _->rw_lk.lock_write(true);
        return *this;
    }

    Handle<T>& ReleaseReadLock(){
        _->rw_lk.unlock_read();
        return *this;
    }

    Handle<T>& ReleaseWriteLock(){
        _->rw_lk.unlock_write();
        return *this;
    }

    Handle<T>& ConvertWriteToReadLock(){
        LOG_DEBUG("call handle wr");
        _->rw_lk.converse_write_to_read(true);
        LOG_DEBUG("call handle wr ok");
        return *this;
    }
    auto GetResourceType() const {
        return _->type;
    }
private:


};

template<typename T, typename... Args>
auto NewHandle(ResourceType access, Args&&... args){
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