#pragma once
#include <cstdint>

#include "../common.hpp"

VUTIL_BEGIN

    uint32_t get_current_thread_index();

    void register_thread_index(uint32_t thread_index);

VUTIL_END