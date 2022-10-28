#include "alloc.hpp"
#ifdef VISER_OS_WIN32
#include <windows.h>
#else

#endif

VUTIL_BEGIN

    size_t get_free_memory_bytes(){
#ifdef VISER_OS_WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        size_t free = status.ullAvailPhys;
        return free;
#else

#endif
    }

VUTIL_END