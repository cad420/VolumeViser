#include "../cuda/kernel.hpp"
#include "../cuda/device.hpp"
#include "../cuda/buffer.hpp"
#include "../cuda/texture.hpp"
#include "../cuda/transfer.hpp"
#include "../cuda/stream.hpp"
#include "../cuda/profiler.hpp"
#include "../cuda/array.hpp"
#include "../cuda/texture.hpp"

int main(){
    try {
        cub::cu_physical_device pd(0);
        std::cout << pd.get_device_name() << std::endl;
        auto ctx = pd.create_context(0);

        auto kernel = ctx.create_kernel();

        auto buffer = ctx.alloc_buffer(1024, cub::e_cu_host);

        auto buffer_view = buffer.view_3d<uint8_t>({}, {});

        auto array = ctx.alloc_array<uint8_t,3>(cub::cu_extent{256, 256, 256});

        auto trans_task = cub::cu_memory_transfer(buffer_view, array, {});

        auto tex = cub::cu_texture(array, {});

        auto task = kernel.pending({},CUB_GPU_LAMBDA(dim3 blockIdx, dim3 threadIdx){
            tex2D<uchar1>(tex.get_handle(), 1.f, 1.f);
            printf("%d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
            return;
        });
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
    }
}