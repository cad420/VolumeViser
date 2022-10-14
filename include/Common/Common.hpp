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

using CUDAVolumeImage = CUDAImage<uint8_t, 3>;

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

using BoundingBox3D = vutil::aabb3f;


// Viser
class GPUMemMgr;
class CPUMemMgr;
class ResourceMgr;
class Renderer;
class DistributeMgr;


class ViserFileOpenError : public std::exception{
public:
    ViserFileOpenError(const std::string& msg) : std::exception(msg.c_str()){}
};


VISER_END