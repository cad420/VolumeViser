#pragma once

#include "common.hpp"

VUTIL_GL_BEGIN

    template<typename T>
    class attrib_var_t{
        GLuint location;
    public:

        explicit attrib_var_t(GLuint location = 0) noexcept
                :location(location)
        {}

        GLuint get_location() const noexcept{
            return location;
        }

    };

VUTIL_GL_END