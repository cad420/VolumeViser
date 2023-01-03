#pragma once

#include <Extension/IOInterface.hpp>

VISER_BEGIN

class EBVolumeFilePrivate;
class EBVolumeFile : public VolumeIOInterface{
public:
    /**
     * @param filename 体数据的desc文件 "*.encoded_blocked.desc.json"
     */
    explicit EBVolumeFile(std::string_view filename);

    ~EBVolumeFile() override;

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    using VolumeDesc = VolumeIOInterface::VolumeDesc;

    VolumeDesc GetVolumeDesc() override;

    /**
     * @param ptr 指向读取数据存储的buffer区域，必须是cpu指针，否则可能有未知错误
     */
    void ReadVolumeRegion(const Int3& beg_pos, const Int3& end_pos, void* ptr) override;

    /**
     * @note 无法写更新，不实现，调用会引发assert false
     */
    void WriteVolumeRegion(const Int3& beg_pos, const Int3& end_pos, const void* ptr) override;

protected:
    std::unique_ptr<EBVolumeFilePrivate> _;
};

class RawVolumeFilePrivate;
class RawVolumeFile : public VolumeIOInterface{
  public:
    /**
     * @param filename 体数据的desc文件 "*.encoded_blocked.desc.json"
     */
    explicit RawVolumeFile(std::string_view filename);

    ~RawVolumeFile() override;

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    using VolumeDesc = VolumeIOInterface::VolumeDesc;

    VolumeDesc GetVolumeDesc() override;

    /**
     * @param ptr 指向读取数据存储的buffer区域，必须是cpu指针，否则可能有未知错误
     */
    void ReadVolumeRegion(const Int3& beg_pos, const Int3& end_pos, void* ptr) override;

    /**
     * @note 无法写更新，不实现，调用会引发assert false
     */
    void WriteVolumeRegion(const Int3& beg_pos, const Int3& end_pos, const void* ptr) override;

  protected:
    std::unique_ptr<RawVolumeFilePrivate> _;
};

VISER_END
