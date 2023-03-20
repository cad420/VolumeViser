#include <Core/Distribute.hpp>



#include <mpi.h>

VISER_BEGIN

class MPI{
  public:
    inline static int world_rank = 0;
    inline static int world_size = 1;
    inline static char process_name[256];
    inline static int process_name_length = 256;

    inline static int root_rank = 0;
};

DistributeMgr::DistributeMgr()
{

}

DistributeMgr::~DistributeMgr()
{

}

void DistributeMgr::Lock()
{

}

void DistributeMgr::UnLock()
{

}

UnifiedRescUID DistributeMgr::GetUID() const
{
    return GenUnifiedRescUID(1, UnifiedRescType::DistributeMgr);
}

void DistributeMgr::WaitForSync()
{

    MPI_Barrier(MPI_COMM_WORLD);
}

int DistributeMgr::GetWorldRank()
{
    return MPI::world_rank;
}

int DistributeMgr::GetWorldSize()
{
    return MPI::world_size;
}

int DistributeMgr::GetRootRank()
{
    return MPI::root_rank;
}

bool DistributeMgr::IsRoot()
{
    return MPI::world_rank == MPI::root_rank;
}

void DistributeMgr::SetRootRank(int rank)
{
    MPI::root_rank = rank;
}

struct _MPI_Init{
    _MPI_Init(){
        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &MPI::world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &MPI::world_size);
        MPI_Get_processor_name(MPI::process_name, &MPI::process_name_length);
        SET_LOG_LEVEL_DEBUG
        LOG_DEBUG("mpi init info: rank({}), size({}), name({})", MPI::world_rank, MPI::world_size, MPI::process_name);
    }
    ~_MPI_Init(){
        MPI_Finalize();
    }
};

static _MPI_Init _init;

DistributeMgr &DistributeMgr::GetInstance()
{
    static DistributeMgr mgr;
    return mgr;
}

void DistributeMgr::Bcast(void *data, int count, int type)
{
    static int table[] = {
      MPI_UNSIGNED_CHAR,
        MPI_CHAR,
        MPI_INT,
        MPI_UNSIGNED,
        MPI_FLOAT,
        MPI_INT64_T,
        MPI_UNSIGNED_LONG_LONG,
        MPI_DOUBLE
    };
    MPI_Bcast(data, count, table[type], GetRootRank(), MPI_COMM_WORLD);
}

VISER_END