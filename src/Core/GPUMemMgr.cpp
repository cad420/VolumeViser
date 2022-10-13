#include <Core/GPUMemMgr.hpp>


VISER_BEGIN

    namespace{
        struct SharedRescWrap{

        };
    }

    class GPUMemMgrPrivate{
    public:
        size_t used_mem_bytes = 0;

        std::mutex mgr_mtx;

        struct SharedResc{

        };

    };


    GPUMemMgr::GPUMemMgr(GPUMemMgrCreateInfo info) {

    }

    GPUMemMgr::~GPUMemMgr() {

    }

    void GPUMemMgr::Lock() {

    }

    void GPUMemMgr::UnLock() {

    }




VISER_END


