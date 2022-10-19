#pragma once

#include "../common.hpp"

#include <string>
#include <vector>

VUTIL_BEGIN

    /**
     * @brief read binary file
     */
    std::vector<unsigned char> read_raw_file_bytes(const std::string& filename);

    /**
     * @brief read file as text
     */
    std::string read_txt_file(const std::string& filename);

    /**
     * @brief write binary context into file
     */
    void write_raw_file_bytes(const std::string& filename,const void* data,size_t bytes_count);



VUTIL_END