#undef UTIL_ENABLE_OPENGL

#include <Algorithm/MeshSmooth.hpp>
#include <Core/GPUMemMgr.hpp>
#include "../Common/helper_math.h"

VISER_BEGIN

namespace {

    struct MeshSmoothingParams{
        // vertex_count index_count
        CUDABufferView1D<uint32_t> mesh_indices;// index_count * sizeof(uint32_t)
        CUDABufferView1D<float3> mesh_vertices_pos;// vertex_count * sizeof(float3)
        CUDABufferView1D<float3> mesh_vertices_normal; // vertex_count * sizeof(float3)
        CUDABufferView1D<uint64_t> vertex_triangle_offset_count; // vertex_count * sizeof(uint64_t)
        //triangle index not vertex index
        CUDABufferView1D<uint32_t> vertex_triangle_index_array;// index_count * sizeof(uint32_t)

        CUDABufferView1D<float3> gen_mesh_vertices_pos;// vertex_count * sizeof(float3)

        int vertex_count = 0;
        int index_count = 0;

        float c = 0.f;
    };


    CUB_KERNEL void RegenerateNormalKernel(MeshSmoothingParams params){
        const unsigned int block_idx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
        const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
        const unsigned int block_vertex_count = blockDim.x * blockDim.y * blockDim.z;
        const unsigned int vertex_index = block_idx * block_vertex_count + thread_idx;
        if(vertex_index >= params.vertex_count) return;

        uint64_t offset_count = params.vertex_triangle_offset_count.at(vertex_index);
        uint32_t offset = offset_count >> 32;
        uint32_t count = offset_count & 0xffffffffu;
        float3 ret{0.f, 0.f, 0.f};
        auto is_valid = [](float3 v){
            bool ok = isfinite(v.x) && !isnan(v.x)
            && isfinite(v.y) && !isnan(v.y)
            && isfinite(v.z) && !isnan(v.z);
            return ok && !(v.x == 0.f && v.y == 0.f && v.z == 0.f);
        };
        for(auto i = 0u; i < count; i++){
            auto triangle_index = params.vertex_triangle_index_array.at(offset + i) * 3;
            auto idx_a = params.mesh_indices.at(triangle_index);
            auto idx_b = params.mesh_indices.at(triangle_index + 1);
            auto idx_c = params.mesh_indices.at(triangle_index + 2);

            auto ab = params.mesh_vertices_pos.at(idx_b) - params.mesh_vertices_pos.at(idx_a);
            auto ac = params.mesh_vertices_pos.at(idx_c) - params.mesh_vertices_pos.at(idx_a);
            auto norm = cross(ab, ac);
            if(is_valid(norm)) ret += norm;
        }
        params.mesh_vertices_normal.at(vertex_index) = is_valid(ret) ? normalize(ret) : ret;
    }


    //一个thread负责一个顶点
    CUB_KERNEL void MeshSmoothingGenKernel(MeshSmoothingParams params){
        const unsigned int block_idx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
        const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
        const unsigned int block_vertex_count = blockDim.x * blockDim.y * blockDim.z;
        const unsigned int vertex_index = block_idx * block_vertex_count + thread_idx;
        if(vertex_index >= params.vertex_count) return;

        float3 ret{0.f, 0.f, 0.f};
        auto vertex_pos = params.mesh_vertices_pos.at(vertex_index);


        uint64_t offset_count = params.vertex_triangle_offset_count.at(vertex_index);
        uint32_t offset = offset_count >> 32;
        uint32_t count = offset_count & 0xffffffffu;

        float3 vsub{0.f, 0.f, 0.f};
        int cnt = 0;

        for (auto i = 0u; i < count; i++)
        {
            auto triangle_index = params.vertex_triangle_index_array.at(offset + i) * 3;

            auto idx_a = params.mesh_indices.at(triangle_index);
            auto idx_b = params.mesh_indices.at(triangle_index + 1);
            auto idx_c = params.mesh_indices.at(triangle_index + 2);

            vsub += params.mesh_vertices_pos.at(idx_a) - vertex_pos;
            vsub += params.mesh_vertices_pos.at(idx_b) - vertex_pos;
            vsub += params.mesh_vertices_pos.at(idx_c) - vertex_pos;
            cnt += 2;
        }
        ret = (1.f / cnt) * vsub;


        params.gen_mesh_vertices_pos.at(vertex_index) = vertex_pos + params.c * ret;
    }

    //一个thread负责一个三角形
    CUB_KERNEL void RemoveNoTriangle(MeshSmoothingParams params){
        const unsigned int block_idx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
        const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
        const unsigned int block_vertex_count = blockDim.x * blockDim.y * blockDim.z;
        const unsigned int triangle_index = (block_idx * block_vertex_count + thread_idx) * 3;
        if(triangle_index >= params.index_count) return;

        auto& idx_a = params.mesh_indices.at(triangle_index);
        auto& idx_b = params.mesh_indices.at(triangle_index + 1);
        auto& idx_c = params.mesh_indices.at(triangle_index + 2);

        auto& A = params.mesh_vertices_pos.at(idx_a);
        auto& B = params.mesh_vertices_pos.at(idx_b);
        auto& C = params.mesh_vertices_pos.at(idx_c);

        auto is_valid = [](float3 v){
            bool ok = isfinite(v.x) && !isnan(v.x)
                      && isfinite(v.y) && !isnan(v.y)
                      && isfinite(v.z) && !isnan(v.z);
            return ok && !(v.x == 0.f && v.y == 0.f && v.z == 0.f);
        };

        auto not_a_triangle = [&A, &B, &C, &is_valid](){
            if(A == B || A == C || B == C) return true;
            auto N = cross(B - A, C - A);
            return !is_valid(N);
        };

        if(not_a_triangle()){
            idx_a = idx_b = idx_c = INVALID_INDEX;
        }
    }

}

class MeshSmootherPrivate{
  public:
    MeshSmoothingParams params;

    CUDAContext ctx;

    CUDAStream stream;

    std::mutex mtx;

    UnifiedRescUID uid;

    Ref<GPUMemMgr> gpu_mem_mgr_ref;

    Ref<HostMemMgr> host_mem_mgr_ref;

    Handle<CUDAHostBuffer> pinned_buffer;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::MeshSmoother);
    }
};

MeshSmoother::MeshSmoother(const MeshSmootherCreateInfo & info)
{
    _ = std::make_unique<MeshSmootherPrivate>();

    _->uid = _->GenRescUID();

    assert(info.gpu_mem_mgr.IsValid());

    _->ctx = info.gpu_mem_mgr->_get_cuda_context();

    _->stream = CUDAStream(_->ctx);

    // whether lock decided by outside
    _->gpu_mem_mgr_ref = info.gpu_mem_mgr;

    _->host_mem_mgr_ref = info.host_mem_mgr;

    _->pinned_buffer = _->host_mem_mgr_ref.Invoke(&HostMemMgr::AllocPinnedHostMem, ResourceType::Buffer, info.reserved_pinned_host_mem_bytes, false);

}

MeshSmoother::~MeshSmoother()
{

}

void MeshSmoother::Lock()
{
    _->mtx.lock();
}

void MeshSmoother::UnLock()
{
    _->mtx.unlock();
}

UnifiedRescUID MeshSmoother::GetUID() const
{
    return _->uid;
}

void MeshSmoother::Smoothing(MeshData0 &mesh, float lambda, float mu, int iterations)
{
    // 先获取每个顶点所属于的三角形数量
    // 扫描累加， 分配一个总和大小的数组， 读取时根据偏移存入实际的所属于三角形情况
    // 然后再根据偏移和数量进行平滑
    int vertex_count = mesh.vertices.size();
    assert(mesh.indices.size() % 3 == 0);
    int index_count = mesh.indices.size();
    int triangle_count = mesh.indices.size() / 3;

    // alloc cuda device buffer
    auto calc_total_gpu_bytes = [vertex_count = static_cast<size_t>(vertex_count), index_count = mesh.indices.size()](){
        size_t mesh_indices_bytes = index_count * sizeof(uint32_t);
        size_t mesh_vertices_pos_bytes = vertex_count * sizeof(float3);
        size_t mesh_vertices_normal_bytes = vertex_count * sizeof(float3);
        size_t vertex_triangle_offset_count = vertex_count * sizeof(uint64_t);
        size_t vertex_triangle_index_array = index_count * sizeof(uint32_t);
        return mesh_indices_bytes + mesh_vertices_pos_bytes * 2 + mesh_vertices_normal_bytes
               + vertex_triangle_offset_count + vertex_triangle_index_array;
    };
    size_t total_gpu_bytes = calc_total_gpu_bytes();
    auto gpu_buffer = _->gpu_mem_mgr_ref->AllocBuffer(ResourceType::Buffer, total_gpu_bytes);


    // copy to pinned buffer
    size_t _offset = 0;
    CUDABufferView1D<uint32_t> mesh_indices = _->pinned_buffer->view_1d<uint32_t>(index_count * sizeof(uint32_t), _offset);
    _->params.mesh_indices = gpu_buffer->view_1d<uint32_t>(index_count * sizeof(uint32_t), _offset);
    _offset += mesh_indices.size();
    CUDABufferView1D<float3> mesh_vertices_pos = _->pinned_buffer->view_1d<float3>(vertex_count * sizeof(float3), _offset);
    _->params.mesh_vertices_pos = gpu_buffer->view_1d<float3>(vertex_count * sizeof(float3), _offset);
    _offset += mesh_vertices_pos.size();
    CUDABufferView1D<float3> mesh_vertices_normal = _->pinned_buffer->view_1d<float3>(vertex_count * sizeof(float3), _offset);
    _->params.mesh_vertices_normal = gpu_buffer->view_1d<float3>(vertex_count * sizeof(float3), _offset);
    _offset += mesh_vertices_normal.size();

    std::memcpy(mesh_indices.data(), mesh.indices.data(), index_count * sizeof(uint32_t));
    for(int i = 0; i < vertex_count; i++){
        auto& pos = mesh.vertices[i].pos;
        mesh_vertices_pos.at(i) = float3{pos.x, pos.y, pos.z};
        auto& normal = mesh.vertices[i].normal;
        mesh_vertices_normal.at(i) = float3{normal.x, normal.y, normal.z};
    }

    auto indices = mesh.indices;
    std::sort(indices.begin(), indices.end());
    indices.push_back(UINT32_MAX);
    assert(indices.size() > 1);
    auto prev = indices.front();
    uint32_t count = 1;
    // 一个顶点属于几个三角形 长度与顶点数相同
    CUDABufferView1D<uint64_t> vertex_triangle_offset_count = _->pinned_buffer->view_1d<uint64_t>(vertex_count * sizeof(uint64_t), _offset);
    _->params.vertex_triangle_offset_count = gpu_buffer->view_1d<uint64_t>(vertex_count * sizeof(uint64_t), _offset);
    _offset += vertex_triangle_offset_count.size();

    size_t vtc_size = 0;
    for(auto it = indices.begin() + 1; it != indices.end(); it++){
        if(*it == prev){
            ++count;
        }
        else{
            assert(prev == vtc_size);
            vertex_triangle_offset_count.at(vtc_size++) = count;
            count = 1;
            prev = *it;
        }
    }
    assert(vtc_size == vertex_count);

    // 前缀和 长度与顶点数相同
    size_t offset = 0;
    for(int i = 0; i < vertex_count; i++){
        auto cnt = vertex_triangle_offset_count.at(i);
        vertex_triangle_offset_count.at(i) |= offset << 32;
        offset +=  cnt;
    }

    // 每个顶点所属的三角形索引数组 数量与索引数量相同
    CUDABufferView1D<uint32_t> vertex_triangle_index_array = _->pinned_buffer->view_1d<uint32_t>(index_count * sizeof(uint32_t), _offset);
    _->params.vertex_triangle_index_array = gpu_buffer->view_1d<uint32_t>(index_count * sizeof(uint32_t), _offset);
    _offset += vertex_triangle_index_array.size();


    std::vector<uint32_t> index_offset(mesh.indices.size());
    for(int i = 0; i < triangle_count; i++){
        auto idx_a = mesh.indices[i * 3];
        auto idx_b = mesh.indices[i * 3 + 1];
        auto idx_c = mesh.indices[i * 3 + 2];

        auto offset_a = (vertex_triangle_offset_count.at(idx_a) >> 32) + index_offset[idx_a]++;
        assert(index_offset[idx_a] <= (vertex_triangle_offset_count.at(idx_a) & 0xffffffff));
        vertex_triangle_index_array.at(offset_a) = i;

        auto offset_b = (vertex_triangle_offset_count.at(idx_b) >> 32) + index_offset[idx_b]++;
        assert(index_offset[idx_b] <= (vertex_triangle_offset_count.at(idx_b) & 0xffffffff));
        vertex_triangle_index_array.at(offset_b) = i;

        auto offset_c = (vertex_triangle_offset_count.at(idx_c) >> 32) + index_offset[idx_c]++;
        assert(index_offset[idx_c] <= (vertex_triangle_offset_count.at(idx_c) & 0xffffffff));
        vertex_triangle_index_array.at(offset_c) = i;

    }

    _->params.gen_mesh_vertices_pos = gpu_buffer->view_1d<float3>(vertex_count * sizeof(float3), _offset);
    _offset += _->params.gen_mesh_vertices_pos.size();
    assert(_offset == total_gpu_bytes);

    try
    {
        //transfer pinned buffer to gpu memory
        cub::memory_transfer_info trans_info;
        trans_info.width_bytes = mesh_indices.size();
        cub::cu_memory_transfer(mesh_indices, _->params.mesh_indices, trans_info)
            .launch(_->stream).check_error_on_throw();

        trans_info.width_bytes = mesh_vertices_pos.size();
        cub::cu_memory_transfer(mesh_vertices_pos, _->params.mesh_vertices_pos, trans_info)
            .launch(_->stream).check_error_on_throw();

        trans_info.width_bytes = mesh_vertices_normal.size();
        cub::cu_memory_transfer(mesh_vertices_normal, _->params.mesh_vertices_normal, trans_info)
            .launch(_->stream).check_error_on_throw();

        trans_info.width_bytes = vertex_triangle_offset_count.size();
        cub::cu_memory_transfer(vertex_triangle_offset_count, _->params.vertex_triangle_offset_count, trans_info)
            .launch(_->stream).check_error_on_throw();

        trans_info.width_bytes = vertex_triangle_index_array.size();
        cub::cu_memory_transfer(vertex_triangle_index_array, _->params.vertex_triangle_index_array, trans_info)
            .launch(_->stream).check_error_on_throw();

        _->params.index_count = index_count;
        _->params.vertex_count = vertex_count;

        void* params[] = {&_->params};

        dim3 tile = {32u, 1u, 1u};
        dim3 grid_dim{
            (vertex_count + tile.x - 1) / tile.x, 1u, 1u
        };
        if(grid_dim.x > 65535u){
            grid_dim.y = grid_dim.x / 32768;
            grid_dim.x = 32768;
        }

        cub::cu_kernel_launch_info info{grid_dim, tile};

        trans_info.width_bytes = _->params.gen_mesh_vertices_pos.size();

        for(int i = 0; i < iterations; i++){
            _->params.c = lambda;
            cub::cu_kernel::pending(info, &MeshSmoothingGenKernel, params)
                .launch(_->stream).check_error_on_throw();

            cub::cu_memory_transfer(_->params.gen_mesh_vertices_pos, _->params.mesh_vertices_pos, trans_info)
                .launch(_->stream).check_error_on_throw();

            _->params.c = mu;
            cub::cu_kernel::pending(info, &MeshSmoothingGenKernel, params)
                .launch(_->stream).check_error_on_throw();

            cub::cu_memory_transfer(_->params.gen_mesh_vertices_pos, _->params.mesh_vertices_pos, trans_info)
                .launch(_->stream).check_error_on_throw();

        }

        cub::cu_kernel::pending(info, &RegenerateNormalKernel, params)
            .launch(_->stream).check_error_on_throw();

        info.grid_dim = dim3{};
        info.grid_dim.x = (triangle_count + tile.x - 1) / tile.x;
        if(grid_dim.x > 65535u){
            grid_dim.y = grid_dim.x / 32768;
            grid_dim.x = 32768;
        }
        cub::cu_kernel::pending(info, &RemoveNoTriangle, params)
            .launch(_->stream).check_error_on_throw();

        // copy from gpu to cpu
        trans_info.width_bytes = mesh_indices.size();
        cub::cu_memory_transfer(_->params.mesh_indices, mesh_indices,trans_info)
            .launch(_->stream).check_error_on_throw();

        trans_info.width_bytes = mesh_vertices_pos.size();
        cub::cu_memory_transfer(_->params.mesh_vertices_pos, mesh_vertices_pos, trans_info)
            .launch(_->stream).check_error_on_throw();

        trans_info.width_bytes = mesh_vertices_normal.size();
        cub::cu_memory_transfer(_->params.mesh_vertices_normal, mesh_vertices_normal, trans_info)
            .launch(_->stream).check_error_on_throw();

        std::memcpy(mesh.indices.data(), mesh_indices.data(), index_count * sizeof(uint32_t));
        for(int i = 0; i < vertex_count; i++){
            auto& pos = mesh_vertices_pos.at(i);
            mesh.vertices[i].pos = Float3(pos.x, pos.y, pos.z);
            auto& normal = mesh_vertices_normal.at(i);
            mesh.vertices[i].normal = Float3(normal.x, normal.y, normal.z);
        }
        LOG_DEBUG("finish gpu mesh smoothing");
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("MeshSmoother Smoothing failed with: {}", err.what());
    }
}

VISER_END