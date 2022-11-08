#pragma once

#include <Extension/IOInterface.hpp>

VISER_BEGIN

// swc描述文件 .json
// 记录该swc的一些信息，比如由哪个体数据生成、与哪些数据块相交、光滑和化简的参数、统计信息
// 包括节点的个数、总的大小、神经元长度、占据体积、表面积
    struct SWCFileDesc{
        std::string filename; // file path for SWCDescFile
        std::string data_path; // file path for SWCFile

        std::string volume_desc_file_path; // 生成SWC的体数据DESC文件


        size_t swc_node_count;
        double swc_total_length;
        size_t swc_total_interior_node;
        double swc_total_volume;
        double swc_total_surface_area;


        SWCFileDesc(std::string_view filename);


    };

// 读取和写入 .swc文件
class SWCFilePrivate;
class SWCFile : public SWCIOInterface{
public:
    static constexpr const char* SWC_FILENAME_EXT_TXT = ".swc";
    static constexpr const char* SWC_FILENAME_EXT_BIN = ".bin";
    struct SWCPoint{
        int id;
        int tag;
        float x;
        float y;
        float z;
        float radius;
        int pid;
        int pad = 0;
    };
    static_assert(sizeof(SWCPoint) == 32, "SWCPoint's size should be power of two or cache line size");

    enum Mode{
        Read,
        Write
    };

    SWCFile();

    ~SWCFile();

    void Lock() override;

    void UnLock() override;

    UnifiedRescUID GetUID() const override;

    void Open(std::string_view filename, Mode mode);

    std::vector<SWCPoint> GetAllPoints() noexcept;

    void WritePoints(int count, std::function<const SWCPoint&(int)> get) noexcept;

    //保存的时候一定要按dfs遍历
    void WritePoints(const std::vector<SWCPoint>& points) noexcept;

    void Close() noexcept;

protected:
    std::unique_ptr<SWCFilePrivate> _;
};



VISER_END