#pragma once

#include <Common/Common.hpp>

VISER_BEGIN

/**
 * @brief 统一的gpu资源管理，提供渲染器需要的资源，如果一部分的资源被一个渲染器使用，那么
 * 这部分的资源无法被其它渲染器使用，如果一个渲染器无法获取足够的资源，那么需要等待。
 * 即对资源的读写锁粒度为一个完整的GPU资源对象，而不是以前的资源对象的某一区域
 * 因此可以有多个渲染器一起使用不冲突、不重合的资源。
 */

int GetGPUCount();

class GPUMemMgrPrivate;
class GPUMemMgr{
public:
    enum RescType{
        Buffer,
        PitchedBuffer,
        Image1D,
        Image2D,
        Image3D
    };

    enum RescAccess{
        Unique,
        Shared
    };

    template<typename T>
    class Handle{
    public:
        struct AccessLocker{
            AccessLocker(std::unique_lock<std::mutex> lk)
            :lk(std::move(lk))
            {}
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
            if(_->access == Unique){
                // always success
                return {std::unique_lock<std::mutex>(_->mtx)};
            }
            else if(_->access == Shared){
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

        Handle(RescAccess access, size_t uid)
        :_(std::make_shared<Inner>())
        {
            _->access = access;
            _->uid = uid;

        }

        ~Handle(){

        }
    private:
        struct Inner{
            std::mutex mtx;
            RescAccess access;
            std::shared_ptr<T> resc;
            size_t uid = 0;
        };
    private:
        std::shared_ptr<Inner> _;
    };

    struct GPUMemMgrCreateInfo{
        int GPUIndex;
        size_t MaxGPUMemBytes;

    };


    explicit GPUMemMgr(GPUMemMgrCreateInfo info);

    ~GPUMemMgr();

    // 整体的加锁
    void Lock();

    void UnLock();

    //创建GPU资源会返回一个句柄
    // create unique resource or shared resource
    // 如果是Shared的资源，Box提供加锁操作，Unique资源总是能加锁成功，而Shared则不一定

    Handle<CUDABuffer> AllocBuffer(RescAccess access);

    Handle<CUDAPitchedBuffer> AllocPitchedBuffer(RescAccess access);

    template<typename T, int N>
    Handle<CUDAImage<T,N>> AllocImage(RescAccess access);

    Handle<CUDAVolumeImage> AllocVolumeImage(RescAccess access);

// 假设有N个未被加锁的纹理，一次acquire需要k个纹理，如果k > N，则排队等待到足够的纹理再处理。
// 每个纹理原先分别有a_i个需要的纹理块，
// 那么进行降序排序后，选择前k个纹理，把其它未被选择/加锁的纹理中的纹理块直接拷贝到被选择的纹理处。
// 对这k个纹理进行加锁。
// 另外每次acquire加一个锁，表示每次只能有一个acquire在处理

protected:
    std::unique_ptr<GPUMemMgrPrivate> _;

};



VISER_END