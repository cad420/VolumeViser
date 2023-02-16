set(MPI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Include
        CACHE STRING "" FORCE)

set(MPI_LIBS
        ${CMAKE_CURRENT_SOURCE_DIR}/Lib/x64/msmpi.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/Lib/x64/msmpifec.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/Lib/x64/msmpifmc.lib
        CACHE STRING "" FORCE)