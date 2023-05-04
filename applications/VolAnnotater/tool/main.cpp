#include <console/cmdline.hpp>

#include <json.hpp>

#define USE_SDF
#define USE_LINEAR_BUFFER_FOR_TEXTURE

#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Algorithm/MarchingCube.hpp>
#include <Algorithm/Voxelization.hpp>
#include <Algorithm/MeshSmooth.hpp>
#include <Core/Renderer.hpp>
#include <Core/HashPageTable.hpp>
#include <Model/SWC.hpp>
#include <Model/Mesh.hpp>
#include <Model/SWC.hpp>
#include <Geometry/GridOctTree.hpp>
#include <IO/SWCIO.hpp>

#include <queue>
#include <unordered_set>
#include <ranges>
#include <fstream>

using namespace viser;
using namespace vutil;

int main(int argc, char** argv){
    struct{
        int gpu_index = 0;
        size_t max_host_mem_bytes = 6ull << 30;
        size_t max_gpu_mem_bytes = 3ull << 30;
        int virtual_texture_count = 2;
        Int3 virtual_texture_shape{1024, 1024, 1024};
    }memory_info;

    struct{
        uint32_t block_length = 126;
        uint32_t padding = 1;
        int lod = 0;
        int smooth_count = 128;
        float smooth_lambda = 0.3;
        float smooth_mu = -0.2;
        int partial_blocks_size = 128;
    }algo_info;

    struct{
        Float3 voxel_space;
        UInt3 voxel_dim;
    }volume_info;

    bool transform = false;
    float transform_ratio = 1.f;

    std::string export_dir = "./";

    std::vector<std::string> input_neurons;

    // reading from json
    {
        cmdline::parser cmd;

        cmd.add<std::string>("config", 'c', "input config json filename", true);

        cmd.parse_check(argc, argv);

        std::string filename = cmd.get<std::string>("config");
        std::ifstream in(filename);
        if(!in.is_open()){
            LOG_ERROR("Failed to open config file: {}", filename.c_str());
            return 0;
        }
        nlohmann::json j;
        in >> j;
        auto must_have = [&](const std::string& item, auto& json)->decltype(auto){
            if(json.count(item) == 0){
                LOG_ERROR("Json config file missed item({})", item);
                exit(0);
            }
            return json.at(item);
        };
        // volume
        {
            auto& volume = must_have("volume", j);
            std::array<float,3> voxel_space = must_have("voxel_space", volume);
            std::array<int, 3> voxel_dim = must_have("voxel_dim", volume);
            volume_info.voxel_space = {voxel_space[0], voxel_space[1], voxel_space[2]};
            volume_info.voxel_dim = UInt3(voxel_dim[0], voxel_dim[1], voxel_dim[2]);
        }
        // input neurons
        {
            std::vector<std::string> neurons = must_have("neurons", j);
            input_neurons = std::move(neurons);
            assert(!input_neurons.empty());
        }

        bool ok;
        nlohmann::json::value_type empty;
        auto may_have = [&](const std::string& item, auto& json)-> nlohmann::json::value_type&{
            if(json.count(item) == 0){
                ok = false;
                return empty;
            }
            ok = true;
            return json.at(item);
        };
        // memory
        {
            auto& memory = may_have("memory", j);
            if(ok) {
                int gpu_idx = may_have("gpu_index", memory); if(ok) memory_info.gpu_index = gpu_idx;
                int max_host_mem_gb = may_have("max_host_mem_gb", memory); if(ok) memory_info.max_host_mem_bytes = size_t(max_host_mem_gb) << 30;
                int max_gpu_mem_gb = may_have("max_gpu_mem_gb", memory); if(ok) memory_info.max_gpu_mem_bytes = size_t(max_gpu_mem_gb) << 30;
                int virtual_texture_count = may_have("virtual_texture_count", memory); if(ok) memory_info.virtual_texture_count = virtual_texture_count;
                std::array<int,3> virtual_texture_shape = may_have("virtual_texture_shape", memory); if(ok) memory_info.virtual_texture_shape = {virtual_texture_shape[0],
                                                          virtual_texture_shape[1], virtual_texture_shape[2]};
            }
        }
        // algo params
        {
            auto& algo = may_have("algo", j);
            if(ok){
                int block_length = may_have("block_length", algo); if(ok) algo_info.block_length = block_length;
                int padding = may_have("padding", algo); if(ok) algo_info.padding = padding;
                int lod = may_have("lod", algo); if(ok) algo_info.lod = lod;
                int smooth_count = may_have("smooth_count", algo); if(ok) algo_info.smooth_count = smooth_count;
                float smooth_lambda = may_have("smooth_lambda", algo); if(ok) algo_info.smooth_lambda = smooth_lambda;
                float smooth_mu = may_have("smooth_mu", algo); if(ok) algo_info.smooth_mu = smooth_mu;
                int partial_blocks_size = may_have("partial_blocks_size", algo); if(ok) algo_info.partial_blocks_size = partial_blocks_size;
            }
        }
        // other
        {
            std::string _export_dir = may_have("export_dir", j);
            if(ok) export_dir = _export_dir;
            if(export_dir.back() == '\\') export_dir.back() = '/';
            else if(export_dir.back() != '/') export_dir += '/';

            std::string _transform = may_have("transform", j);
            if(ok) transform = _transform == "yes";
            float _transform_ratio = may_have("transform_ratio", j);
            if(ok) transform_ratio = _transform_ratio;
        }
    }

    UInt3 block_dim = (volume_info.voxel_dim + algo_info.block_length - 1u) / algo_info.block_length;
    Float3 block_length_space = volume_info.voxel_space * (float)algo_info.block_length;
    Float3 padding_space = (float)algo_info.padding * volume_info.voxel_space;
    BoundingBox3D volume_bound = {Float3(0.f),
                                  Float3(volume_info.voxel_dim.x * volume_info.voxel_space.x,
                                         volume_info.voxel_dim.y * volume_info.voxel_space.y,
                                         volume_info.voxel_dim.z * volume_info.voxel_space.z)};



    // resource
    auto& resc_ins = ResourceMgr::GetInstance();
    auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                          .MaxMemBytes = memory_info.max_host_mem_bytes,
                                                          .DeviceIndex = -1});

    auto host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

    // build oct tree
    GridOctTree::OctTreeCreateInfo tree_info{
        .host_mem_mgr_ref = host_mem_mgr_ref,
        .leaf_node_shape = block_length_space,
        .world_origin = volume_bound.low,
        .world_range = volume_bound.high,
        .expand_boundary = padding_space,
        .leaf_is_valid = false
    };
#ifdef USE_TREE
    auto oct_tree = NewHandle<GridOctTree>(ResourceType::Object, tree_info);
#endif
    auto gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                             .MaxMemBytes = memory_info.max_gpu_mem_bytes,
                                                             .DeviceIndex = memory_info.gpu_index});

    auto gpu_mem_mgr_ref = resc_ins.GetGPURef(gpu_resc_uid);
    gpu_mem_mgr_ref->_get_cuda_context()->set_ctx();

    GPUVTexMgr::GPUVTexMgrCreateInfo vtex_info{
        .gpu_mem_mgr = gpu_mem_mgr_ref,
        .host_mem_mgr = host_mem_mgr_ref,
        .vtex_count = memory_info.virtual_texture_count,
        .vtex_shape = memory_info.virtual_texture_shape,
        .bits_per_sample = 8,
        .samples_per_channel = 1,
        .vtex_block_length = int(algo_info.block_length + algo_info.padding * 2),
        .is_float = false,
        .exclusive = true
    };
    auto vtex_uid = gpu_mem_mgr_ref->RegisterGPUVTexMgr(vtex_info);

    auto vtex_ref = gpu_mem_mgr_ref->GetGPUVTexMgrRef(vtex_uid);

    auto pt_ref = vtex_ref->GetGPUPageTableMgrRef();

    SWCVoxelizer::VoxelizerCreateInfo voxelizer_info{
        .gpu_mem_mgr = gpu_mem_mgr_ref,
        .host_mem_mgr = host_mem_mgr_ref,
    };
    auto swc_voxelizer = NewHandle<SWCVoxelizer>(ResourceType::Object, voxelizer_info);


    auto segment_buffer = host_mem_mgr_ref->AllocPinnedHostMem(
                                               ResourceType::Buffer,
                                               MaxSegmentCount * sizeof(SWCSegment),
                                               false);

    MarchingCubeAlgo::MarchingCubeAlgoCreateInfo mc_info{
        .gpu_mem_mgr = gpu_mem_mgr_ref,
        .host_mem_mgr = host_mem_mgr_ref
    };

    auto mc_algo = NewHandle<MarchingCubeAlgo>(ResourceType::Object, mc_info);

    VolumeParams vol_params{
        .bound = volume_bound,
        .block_length = algo_info.block_length,
        .padding = algo_info.padding,
        .voxel_dim = volume_info.voxel_dim,
        .space = volume_info.voxel_space
    };

    swc_voxelizer->SetVolume(vol_params);

    mc_algo->SetVolume(vol_params);


#ifndef USE_LINEAR_BUFFER_FOR_TEXTURE
    auto vtexs = vtex_ref->GetAllTextures();
    for(auto& [unit, handle] : vtexs){
        swc_voxelizer->BindVTexture(handle, unit);

        mc_algo->BindVTexture(handle, unit);
    }
#else
    auto vbufs = vtex_ref->GetAllTextureBuffers();
    for(auto& [unit, view] : vbufs){
        swc_voxelizer->BindVBuffer(view, unit);

        mc_algo->BindVBuffer(view, unit);
    }
#endif

    MeshSmoother::MeshSmootherCreateInfo ms_info{
        .gpu_mem_mgr = gpu_mem_mgr_ref.LockRef(),
        .host_mem_mgr = host_mem_mgr_ref.LockRef()
    };

    auto mesh_smoother = NewHandle<MeshSmoother>(ResourceType::Object, ms_info);


    struct{
        std::thread worker;
        std::queue<std::function<void()>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
        std::atomic<bool> stop = false;
        void append(std::function<void()> task){
            std::lock_guard<std::mutex> lk(mtx);
            tasks.push(std::move(task));
            cv.notify_one();
        }
    }mesh_postprocess_queue;

    mesh_postprocess_queue.worker = std::thread([&](){
        while(true){
            std::unique_lock<std::mutex> lk(mesh_postprocess_queue.mtx);
            mesh_postprocess_queue.cv.wait(lk, [&]{
                return !mesh_postprocess_queue.tasks.empty() || mesh_postprocess_queue.stop;
            });
            if(mesh_postprocess_queue.stop && mesh_postprocess_queue.tasks.empty()) break;

            if(mesh_postprocess_queue.tasks.empty()) continue;

            auto task = mesh_postprocess_queue.tasks.front();
            mesh_postprocess_queue.tasks.pop();

            lk.unlock();

            if(task)
                task();

        }
    });

    std::vector<std::pair<std::string,Handle<SWC>>> swcs;

    auto extract_name = [](const std::string& _filename)->std::string{
        auto filename = _filename;
        vutil::replace(filename, '\\', '/');
        auto ret = vutil::split(filename, "/");
        auto name = ret.back();
        ret = vutil::split(name, ".");
        return ret.front();
    };

    SWCFile swc_file;
    auto load_swc = [&](const std::string& filename){
        try{
            swc_file.Open(filename.c_str(), SWCFile::Read);
        }
        catch (const std::exception& err){
            LOG_ERROR("Open swc file({}) failed: ", filename.c_str(), err.what());
            return;
        }
        auto name = extract_name(filename);
        auto& swc = swcs.emplace_back();
        swc.first = export_dir + name + ".obj";
        swc.second = NewHandle<SWC>(ResourceType::Object);
        auto swc_pts = swc_file.GetAllPoints();
        if(transform){
            auto space = volume_info.voxel_space * transform_ratio;
            auto _r = space.length();
            for(auto& pt : swc_pts){
                pt.x *= space.x;
                pt.y *= space.y;
                pt.z *= space.z;
                pt.radius *= _r;
            }
        }
        swc_file.Close();
        std::sort(swc_pts.begin(), swc_pts.end(), [](const auto& a, const auto& b){
            return a.id < b.id;
        });
        for(auto& pt : swc_pts) swc.second->InsertNodeLeaf(pt);
        LOG_INFO("load swc: {} ok, pts count: {}", filename.c_str(), swc_pts.size());

        // calc swc length
        {
            float total_length = 0.f;
            for(auto& pt : swc_pts){
                if(swc.second->QueryNode(pt.pid)){
                    auto& ppt = swc.second->GetNode(pt.pid);
                    auto l = Float3(pt.x - ppt.x, pt.y - ppt.y, pt.z - ppt.z);
                    total_length += l.length();
                }
            }
            LOG_INFO("swc total length: {}", total_length);
        }

    };
    LOG_INFO("start loading neuron swc files...");
    for(auto& neuron : input_neurons) load_swc(neuron);
    LOG_INFO("finish loading neuron swc files, count is : {}", swcs.size());

    using BlockUID = GridVolume::BlockUID;
    using SWCPoint = SWC::SWCPoint;
    using SWCPointKey = SWC::SWCPointKey;

    vutil::thread_group_t tg;
    tg.start(actual_worker_count(18));

    auto swc2mesh_task = [&](Handle<SWC> swc, std::string filename){
        LOG_INFO("start swc to mesh task");
        auto mesh = NewHandle<Mesh>(ResourceType::Object);
        int idx = 0;

        {
            AutoTimer timer("swc2mesh");
            auto get_box = [](const SWC::SWCPoint& pt){
                BoundingBox3D box;
                box |= Float3(pt.x - pt.radius, pt.y - pt.radius, pt.z - pt.radius);
                box |= Float3(pt.x + pt.radius, pt.y + pt.radius, pt.z + pt.radius);
                return box;
            };

            auto pts = swc->PackAll();
            // calc all blocks that swc segments intersect with
            std::unordered_set<BlockUID> swc_intersect_blocks;
            std::unordered_map<SWCPointKey, SWCPoint> pts_mp;
            std::vector<BlockUID> tmp;
            for(auto& pt : pts) pts_mp[pt.id] = pt;
            for(auto& pt : pts){
                //todo replace with octree
                BoundingBox3D box;
                if(pts_mp.count(pt.pid)) box |= get_box(pts_mp.at(pt.pid));

                box |= get_box(pt);
                box.low -= 2.f * padding_space;
                box.high += 2.f * padding_space;

                ComputeIntersectedBlocksWithBoundingBox(tmp,
                                                        block_length_space,
                                                        block_dim,
                                                        volume_bound,
                                                        box);
                for(auto& b : tmp) swc_intersect_blocks.insert(b);
                tmp.clear();
            }

            // partial into small sets
            auto set_task = [&](auto beg, auto end){
                std::vector<BlockUID> partial_blocks;
                std::vector<GridOctTree::NodeIndex> node_indices;
                for(auto it = beg; it != end; it++){
                    const auto& uid = partial_blocks.emplace_back(*it);
                    node_indices.push_back({(int)uid.x, (int)uid.y, (int)uid.z, uid.GetLOD()});
                }
#ifdef USE_TREE
                oct_tree->Set(node_indices, false);
#endif
                auto intersect = [&](const auto& box){
#ifdef USE_TREE
                    return oct_tree->TestIntersect(box);
#else
                    return true;
#endif
                };

                std::vector<SWCSegment> swc_segments;
                std::mutex mtx;
                // calc swc segments for the set of blocks
                int pt_cnt = pts.size();
                vutil::parallel_forrange(0, pt_cnt, [&](int threadIdx, int idx){
                        auto& pt = pts.at(idx);
                        if(pts_mp.count(pt.pid) == 0) return;
                        auto& prev_pt = pts_mp.at(pt.pid);
                        auto box = get_box(prev_pt) | get_box(pt);
                        if(intersect(box)){
                            std::lock_guard<std::mutex> lk(mtx);
                            swc_segments.emplace_back(Float4(prev_pt.x, prev_pt.y, prev_pt.z, prev_pt.radius),
                                                      Float4(pt.x, pt.y, pt.z, pt.radius));
                        }
                    }, tg);


                size_t swc_seg_count = swc_segments.size();
                if(swc_seg_count > MaxSegmentCount){
                    LOG_ERROR("SWC segments count({}) > MaxSegmentCount({})", swc_seg_count, MaxSegmentCount);
                    return;
                }
                LOG_INFO("partial blocks intersect swc segment count: {}", swc_seg_count);


                std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
                pt_ref->GetAndLock(partial_blocks, blocks_info);
                for(auto& key : partial_blocks) pt_ref->Promote(key);

                swc_voxelizer->BindPTBuffer(pt_ref->GetPageTable(true).GetHandle());

                SWCVoxelizer::SWCVoxelizeAlgoParams vparams;
                vparams.ptrs = segment_buffer->view_1d<SWCSegment>(swc_seg_count);
                for(size_t i = 0; i < swc_seg_count; i++)
                    vparams.ptrs.at(i) = swc_segments[i];

                {
                    AutoTimer _t("Voxelizing");
                    swc_voxelizer->Run(vparams);
                }

                mc_algo->BindPTBuffer(pt_ref->GetPageTable(false).GetHandle());

                MarchingCubeAlgo::MarchingCubeAlgoParams mc_params;
                mc_params.shape = UInt3(algo_info.block_length);
#ifdef USE_SDF
//                mc_params.isovalue = 0.0f;
                mc_params.isovalue = vol_params.space.length() * 0.2f;
#else
                mc_params.isovalue = 0.5f;
#endif
                for(auto& b : partial_blocks){
                    mc_params.origin = UInt3(b.x, b.y, b.z) * mc_params.shape;
                    mc_params.lod = b.GetLOD();

                    int gen_tri_num = mc_algo->Run(mc_params);

                    LOG_INFO("gen tri num : {}", gen_tri_num);

                    mesh->Insert(MeshData0(gen_tri_num, [&](int vert_idx)->const Float3&{
                        return mc_params.gen_host_vertices_ret.at(vert_idx);
                    }), idx++);
                }

                pt_ref->Release(partial_blocks);

                //clear page table
                pt_ref->Reset();

                for(auto& [b, coord] : blocks_info)
                    vtex_ref->Clear(b.ToUnifiedRescUID(), coord);
            };

            for(auto& b : swc_intersect_blocks) tmp.emplace_back(BlockUID(b).SetSWC());
            // sort?
            auto all_blocks_count = tmp.size();
            LOG_INFO("mesh all blocks count: {}", all_blocks_count);
            for(size_t offset = 0; offset < all_blocks_count; offset += algo_info.partial_blocks_size){
                auto beg = tmp.begin() + offset;
                auto end = tmp.begin() +
                               ((offset + algo_info.partial_blocks_size) >= all_blocks_count
                                ? all_blocks_count : offset + algo_info.partial_blocks_size);
                set_task(beg, end);
                LOG_INFO("mesh generate from {} to {} ok...", offset, (std::min)(all_blocks_count, offset + algo_info.partial_blocks_size));
            }
            LOG_INFO("mesh generate ok...");
        }


        mesh_postprocess_queue.append([&, mesh = std::move(mesh), filename = std::move(filename)] () mutable {
//            mesh->Smooth(algo_info.smooth_lambda, algo_info.smooth_mu, algo_info.smooth_count);
            {
                AutoTimer timer("smoothing");
                mesh_smoother->Smoothing(mesh->GetPackedMeshDataRef(), algo_info.smooth_lambda, algo_info.smooth_mu,
                                         algo_info.smooth_count);
            }

            MeshFile mesh_file;
            mesh_file.Open(filename, MeshFile::Write);
            mesh_file.WriteMeshData(mesh->GetPackedMeshDataRef());
            mesh_file.Close();
            LOG_INFO("export mesh file: {} ok", filename.c_str());
        });
    };

    for(auto& [export_filename, swc] : swcs){
        //one by one
        swc2mesh_task(std::move(swc), std::move(export_filename));
    }

    mesh_postprocess_queue.stop = true;

    mesh_postprocess_queue.worker.join();

    LOG_INFO("Run all tasks ok...");
}