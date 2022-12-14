#pragma once

#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Algorithm/MarchingCube.hpp>
#include <Algorithm/Voxelization.hpp>
#include <Core/Renderer.hpp>
#include <Core/HashPageTable.hpp>
#include <Model/SWC.hpp>
#include <Model/Mesh.hpp>
#include <Model/SWC.hpp>
#include <IO/SWCIO.hpp>

#include <cuda_gl_interop.h>
#include <json.hpp>

#include <set>
#include <fstream>
#include <unordered_set>

using namespace viser;
using namespace vutil;
using namespace vutil::gl;

using PatchID = size_t;
using BlockUID = GridVolume::BlockUID;

class SWCRenderer;
class NeuronRenderer;

struct VolAnnotaterCreateInfo;

struct AppSettings{
public:
    inline static size_t MaxHostMemBytes = 0;
    inline static size_t MaxRenderGPUMemBytes = 0;
    inline static size_t MaxComputeGPUMemBytes = 0;
    inline static int RenderGPUIndex = 0;
    inline static int ComputeGPUIndex = 0;
    inline static size_t MaxFixedHostMemBytes = 0;
    inline static int ThreadsGroupWorkerCount = 0;
    inline static int VTexCount = 0;
    inline static Int3 VTexShape = Int3(0, 0, 0);

    static void Initialize(const VolAnnotaterCreateInfo& info);
};

struct ViserRescPack{
public:
    Ref<HostMemMgr> host_mem_mgr_ref;

    Ref<GPUMemMgr> render_gpu_mem_mgr_ref;

    Ref<GPUMemMgr> compute_gpu_mem_mgr_ref;

    Ref<FixedHostMemMgr> host_block_pool_ref;

    Ref<GPUVTexMgr> gpu_vtex_mgr_ref;

    Ref<GPUPageTableMgr> gpu_pt_mgr_ref;

    vutil::thread_group_t thread_group;

    struct {
        Handle<GridVolume> volume;

        int max_lod = 0;

        Float3 volume_space_ratio = {1.f, 1.f, 1.f};

        uint32_t block_length = 0;

    }vol_priv_data;

public:
    void Initialize();

    void LoadVolume(const std::string& filename);
private:
    /**
     * @brief 初始化一些与体数据相关的资源，在体数据加载后调用
     */
    void InitializeVolumeResc();
};

/**
 * @brief 大规模体绘制相关资源
 */
struct VolRenderRescPack{
public:

    Handle<CRTVolumeRenderer> crt_vol_renderer;

    float render_base_space = 0.f;

    Handle<FrameBuffer> framebuffer;

    viser::LevelOfDist lod;
    float lod_ratio = 1.f;

    struct{
        Float3 lod0_block_length_space;
        BoundingBox3D volume_bound;
        UInt3 lod0_block_dim;
    }render_vol;

    struct {
        std::vector<GridVolume::BlockUID> intersect_blocks;
        std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> host_blocks;//在循环结束会释放Handle
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> missed_host_blocks;
        std::map<int, std::vector<std::function<void()>>> task_mp;
        std::map<int, vutil::task_group_handle_t> task_groups;
        std::vector<int> lods;

        PerFrameParams per_frame_params;
    }vol_render_priv_data;

    struct {
        vec2i query_pos = {-1, -1};
        bool clicked = false;
        Handle<CUDAHostBuffer> query_info;
        CUDABufferView1D<float> query_info_view;
    }vol_query_priv_data;

public:
    void Initialize(ViserRescPack& _);

    void OnVolumeLoaded(ViserRescPack& _);

    void UpdateUpBoundLOD(ViserRescPack& _, float fov_rad, float ratio = 1.f);

    void UpdateDefaultLOD(ViserRescPack& _, float ratio = 1.f);

    void UpdateTransferFunc(const std::vector<std::pair<float, Float4>>& pts);

    std::vector<BlockUID> ComputeIntersectBlocks(const std::vector<SWC::SWCPoint>& pts);
};

using SWCUID = viser::UnifiedRescUID;
using SWCNeuronID = size_t;
using SWCPointKey = SWC::SWCPointKey;

using MeshUID = viser::UnifiedRescUID;

constexpr size_t SWC_MAX_NEURON_NUM = 1024ull;
constexpr size_t SWC_MAX_POINT_NUM = 1ull << 16;

struct DrawTreeNode;
struct DrawPoint{
    Float2 pos;
    int draw_id = -1;
    SWCPointKey pt_id;
};
struct SWCDrawTree{
public:
    DrawTreeNode* draw_tree_root = nullptr;

    std::vector<std::pair<Float2, Float2>> draw_lines;

    std::vector<DrawPoint> draw_points;

    std::unordered_set<SWCPointKey> draw_segment_points;

    Handle<SWC> swc;
public:

    void Build(Handle<SWC> swc);

    //计算两个点之间的所有点，不包括端点
    void SelSegPoints(SWCPointKey a, SWCPointKey b);

    bool PointInsideSeg(SWCPointKey a) { return draw_segment_points.count(a);}

private:
    //只用于画线的信息，丢失了SWCPoint相关的信息
    std::vector<std::pair<Float2, Float2>> CalcDrawLines();

    //SWCPoint映射到平面上的点信息，用于辅助点选中
    std::vector<DrawPoint> CalcDrawSWCPoints();

};

/**
 * @brief SWC标注和渲染相关资源
 */
struct SWCRescPack{
public:
    Handle<SWCFile> swc_file;
    struct SWCInfo{
        Handle<SWC> swc;
        std::string name;//swc名称
        std::string filename;//文件名
    };
    std::unordered_map<SWCUID, SWCInfo> loaded_swc;//所有加载、创建的swc对象
    SWCUID selected_swc_uid = INVALID_RESC_ID;//当前被选中编辑的swc uid

    //由于mesh也可以自己生成或者从文件中加载 因此两者的对应关系是 swc->mesh but mesh x-> swc
    //因此当切换swc时 也可以切换对应的mesh 但是切换mesh不会改变当前选择的swc
    //每次新建或者加载一个swc时 会新建一个关联的mesh 但是删除swc不会删除对应的mesh
    std::unordered_map<SWCUID, MeshUID> swc_mesh_mp;//swc与对应生成的mesh
    std::function<void(MeshUID)> on_swc_selected;

    // 一个swc可能含有多条神经 每条神经是一颗树 只有一个根节点 并且每条神经之间是不连通的
    struct{

        // 没被使用的neuron id集合
        std::set<SWCNeuronID> available_neuron_ids;

        // 一个swc里的不同neuron共用一个可用的swc点集合
        std::set<SWCPointKey> available_swc_pt_ids;

        //每条神经元根节点到神经元id的mapping
        std::unordered_map<SWCPointKey, SWCNeuronID> pt_to_neuron_mp;

        SWCPointKey last_picked_swc_pt_id = SWC::INVALID_SWC_KEY;

        //当swc被更新时需要重新build
        SWCDrawTree swc_draw_tree;

        int picked_count = 1;
        std::queue<SWCPointKey> picked_swc_pt_q;

        void Reset(){
            available_neuron_ids.clear();
            for(size_t i = 1; i <= SWC_MAX_NEURON_NUM; i++){
                available_neuron_ids.insert(i);
            }

            available_swc_pt_ids.clear();
            for(size_t i = 1; i <= SWC_MAX_POINT_NUM; i++){
                available_swc_pt_ids.insert(i);
            }
            pt_to_neuron_mp.clear();

            last_picked_swc_pt_id = SWC::INVALID_SWC_KEY;

            while(!picked_swc_pt_q.empty()) picked_swc_pt_q.pop();
            for(int i = 0; i < picked_count; i++) picked_swc_pt_q.push(0);
        }
    }swc_priv_data;

    std::unique_ptr<SWCRenderer> swc_renderer;

public:
    bool Selected() const { return CheckUnifiedRescUID(selected_swc_uid) && loaded_swc.count(selected_swc_uid); }

    SWCInfo& GetSelected() { return loaded_swc.at(selected_swc_uid); }

    /**
     * @brief 记录一个被选中的swc point id，自动保持最近2个被选中的
     * 会先检查是否合法的id 如果不合法 则不加入 不抛出异常
     */
    void AddPickedSWCPoint(SWCPointKey id);

    void UpdatePickedSWCSegmentPoints();

    void SetSWCPointPickSize(int s);

    void Initialize();
    /**
     * @brief 从swc文件中加载swc数据到内存模型中
     * @note 完成后不会被切换到选中状态
     */
    void LoadSWCFile(const std::string& filename);

    /**
     * @brief 创建一个新的swc文件
     * @note 成功后会调用SelectSWC
     */
    void CreateSWC(const std::string& filename = "");

    /**
     * @brief 将mesh uid绑定到当前选择的swc
     */
    void BindMeshToSWC(MeshUID mesh_id);

    /**
     * @brief 将当前swc绑定的mesh uid删除
     */
    void DeleteSWCMesh();

    /**
     * @brief 在内存中删除当前选中的swc文件，切换到None
     * 会调用DeleteSWCMesh删除与其绑定的mesh uid
     */
    void DeleteSelSWC();

    /**
     * @brief 切换当前选择的swc文件 会改变swc_priv_data内部成员变量
     */
    void SelectSWC(SWCUID swc_id);

    /**
     * @brief 插入一个点到当前选中的swc
     * @note 传入的点不需要id和pid 内部会生成这两个
     */
    void InsertSWCPoint(SWC::SWCPoint pt);

    /**
     * @brief 将从文件中加载的swc写会文件 如果当前选择的swc没有设置过filename 那么直接返回
     */
    void SaveSWCToFile();

    void Commit();

    /**
     * @brief 将当前选定的swc保存到文件中 文件名包含了swc文件的格式时txt还是bin
     */
    void ExportSWCToFile(const std::string& filename);

private:

};


/**
 * @brief SWC转换为Mesh和神经元网格渲染相关资源
 */
struct SWC2MeshRescPack{
public:
    Handle<MeshFile> mesh_file;
    //一个完整的网格文件 由一个swc生成 可能包含多个不连通的神经元
    struct MeshInfo{
        Handle<Mesh> mesh;
        std::string name;
        std::string filename;
    };
    //所有加载的mesh文件
    std::unordered_map<MeshUID, MeshInfo> loaded_mesh;
    MeshUID selected_mesh_uid;
    // 一个block对应的mesh 这里的PatchID同时也被渲染器使用
    //不同的状态只会影响mesh渲染的对象
    //如果是none 那么什么都不渲染
    //如果是merged 那么渲染器会渲染loaded_mesh里面保存的一个mesh
    //如果是blocked 那么渲染器会渲染patch_mesh_mp里保存的多个mesh
    enum MeshStatus : int {
        None = 0,
        Merged = 1,
        Blocked = 2,
    };

    MeshStatus mesh_status = None;

    enum BlockMeshStatus :int {
        Empty = 0, // mesh没有生成
        Modified = 1, // block涉及到swc的改动
        Updated = 2 // 重新生成了mesh
    };

    struct BlockMesh{
        Handle<Mesh> mesh;
        BlockMeshStatus status = Empty;
        bool visible = true;
    };


    //这里应该存储所有swc涉及范围数据块对应的mesh
    struct{
        //这里的mesh不是完整的 是一个数据块对应的mesh
        //注意这些数据一旦被切换了就会丢失
        std::unordered_map<BlockUID, BlockMesh> patch_mesh_mp;

        Handle<CUDAHostBuffer> segment_buffer;

    }s2m_priv_data;

    Handle<SWCVoxelizer> swc_voxelizer;

    Handle<MarchingCubeAlgo> mc_algo;

    std::unique_ptr<NeuronRenderer> neuron_renderer;

public:
    void Initialize(ViserRescPack& _);

    bool LocalUpdating() const { return mesh_status == Blocked; }

    bool DrawMesh() const { return mesh_status != None; }

    bool QueryBlockMesh(const BlockUID& uid) const { return s2m_priv_data.patch_mesh_mp.count(uid) != 0;}

    void OnVolumeLoaded(ViserRescPack& _, VolRenderRescPack& __);

    /**
     * @brief 创建一个block mesh，并且设置状态为Empty，如果原来存在会被清除
     */
    void CreateBlockMesh(const BlockUID& uid);

    /**
     * @brief 更新一个block mesh，并且设置状态为Updated，如果原来不存在，会先调用CreateBlockMesh创建
     * @note 不会调用MeshUpdated 因为可能有多个block mesh一起更新 需要最后手动调用
     */
    void UpdateBlockMesh(const BlockUID& uid, Handle<Mesh> mesh);

    void UpdateAllBlockMesh();

    /**
     * @note 最后会调用MeshUpdated
     */
    void UpdateMesh(MeshUID uid, Handle<Mesh> mesh);

    void UpdateMesh(Handle<Mesh> mesh) { if(Selected()) UpdateMesh(selected_mesh_uid, std::move(mesh)); }

    /**
     * @brief 将当前Merged状态下的Mesh进行光滑操作，完成后会通知渲染器更新
     * @note 一定要在Selected和Merged状态下
     */
    void SmoothMesh(float lambda, float mu, int iterations);

    /**
     * @brief 将uid指定的block状态改为 status 如果这个block不存在 那么会调用CreateBlockMesh先创建
     */
    void SetBlockMeshStatus(const BlockUID& uid, BlockMeshStatus status);

    /**
     *@brief 将所有patch_mesh_mp里保存的mesh进行合并并保存到loaded_mesh里 会将MeshStatus设置为Merged
     * 最后会调用MeshUpdated
     */
    void MergeAllBlockMesh();

    void SetMeshStatus(MeshStatus status);

    /**
     * @brief 通知mesh渲染器 数据需要更新 根据当前的MeshStatus重新加载数据到渲染器
     */
    void MeshUpdated();

    /**
     * @brief 从文件中加载mesh到内存中，mesh对应uid 设置MeshStatus为Merged，即调用Select
     * @params uid 必须是有效的 即与BlockUID对应
     */
    void LoadMeshFile(const std::string& filename);

    void CreateMesh(const std::string& name = "", const std::string& filename = "");

    bool Selected() const { return CheckUnifiedRescUID(selected_mesh_uid) && loaded_mesh.count(selected_mesh_uid); }

    /**
     * @brief 切换当前选中的mesh 会将MeshStatus设置为Merged，并且会丢失patch_mesh_mp里的数据
     */
    void Select(MeshUID mesh_id);

    MeshInfo& GetSelected() { assert(Selected()); return loaded_mesh.at(selected_mesh_uid); }

    /**
     * @brief 清除patch_mesh_mp里的数据
     */
    void ResetBlockedMesh();

    void SaveMeshToFile();

    void ExportMeshToFile(const std::string& filename);
};

inline std::string ToString(SWC2MeshRescPack::BlockMeshStatus status){
    if(status == SWC2MeshRescPack::Empty)
        return "Empty";
    else if(status == SWC2MeshRescPack::Modified)
        return "Modified";
    else if(status == SWC2MeshRescPack::Updated)
        return "Updated";
    else{
        assert(false);
    }
}



