#pragma once

#include <glad/glad.h>
#if !(defined(__gl_h_) || defined(__GL_H__))
#error "should include opengl header before this"
#endif

#include "../math.hpp"
#include "console/logger.hpp"
#include "misc/concept.hpp"


#define VUTIL_GL_BEGIN namespace vutil::gl{
#define VUTIL_GL_END }

VUTIL_GL_BEGIN



    using vec2 = vec2f;
    using vec3 = vec3f;
    using vec4 = vec4f;

    using mat3 = mat3f_c;
    using mat4 = mat4f_c;

    using transform = mat4::right_transform;

    inline std::string GetGLErrorStr(GLenum gl_error)
    {
        std::string error;
        switch (gl_error)
        {
            case GL_INVALID_ENUM:
                error = "GL_INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                error = "GL_INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                error = "GL_INVALID_OPERATION";
                break;
            case GL_STACK_OVERFLOW:
                error = "GL_STACK_OVERFLOW";
                break;
            case GL_STACK_UNDERFLOW:
                error = "GL_STACK_UNDERFLOW";
                break;
            case GL_OUT_OF_MEMORY:
                error = "GL_OUT_OF_MEMORY";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error = "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            case GL_INVALID_INDEX:
                error = "GL_INVALID_INDEX";
                break;
            default:
                error = "UNKNOWN_ERROR";
                break;
        }
        return error;
    }

    inline void PrintGLErrorType(GLenum gl_error)
    {
        LOG_ERROR("{}", GetGLErrorStr(gl_error).c_str());
    }

    inline GLenum PrintGLErrorMsg(const char *file, int line)
    {
        GLenum error_code;
        while ((error_code = glGetError()) != GL_NO_ERROR)
        {
            std::string error;
            switch (error_code)
            {
                case GL_INVALID_ENUM:
                    error = "GL_INVALID_ENUM";
                    break;
                case GL_INVALID_VALUE:
                    error = "GL_INVALID_VALUE";
                    break;
                case GL_INVALID_OPERATION:
                    error = "GL_INVALID_OPERATION";
                    break;
                case GL_STACK_OVERFLOW:
                    error = "GL_STACK_OVERFLOW";
                    break;
                case GL_STACK_UNDERFLOW:
                    error = "GL_STACK_UNDERFLOW";
                    break;
                case GL_OUT_OF_MEMORY:
                    error = "GL_OUT_OF_MEMORY";
                    break;
                case GL_INVALID_FRAMEBUFFER_OPERATION:
                    error = "GL_INVALID_FRAMEBUFFER_OPERATION";
                    break;
                case GL_INVALID_INDEX:
                    error = "GL_INVALID_INDEX";
                    break;
            }
            LOG_ERROR("{} at line {} in file {}", error.c_str(), line, file);
        }
        return error_code;
    }

#ifdef NDEBUG
    #define GL_REPORT void(0);
#define GL_ASSERT void(0);
#define GL_EXPR(expr) expr;
#define GL_CHECK void(0);
#else
#define GL_REPORT PrintGLErrorMsg(__FILE__, __LINE__);
#define GL_ASSERT assert(glGetError() == GL_NO_ERROR);

#define GL_EXPR(expr)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        GLenum gl_error;                                                                                               \
        int __count = 0;                                                                                                 \
        while ((gl_error = glGetError()) != GL_NO_ERROR)                                                               \
        {                                                                                                              \
            LOG_ERROR("GL error {} before call {} at line {} in file {}", GetGLErrorStr(gl_error).c_str(), #expr, __LINE__,    \
                      __FILE__);                                                                                       \
            __count++;                                                                                                   \
            if (__count > 10)                                                                                            \
                break;                                                                                                 \
        }                                                                                                              \
        expr;                                                                                                          \
        __count = 0;                                                                                                     \
        while ((gl_error = glGetError()) != GL_NO_ERROR)                                                               \
        {                                                                                                              \
            LOG_ERROR("Calling {} caused GL error {} at line {} in file {}", #expr, GetGLErrorStr(gl_error).c_str(), __LINE__, \
                      __FILE__);                                                                                       \
            if (++__count > 10)                                                                                          \
                break;                                                                                                 \
        }                                                                                                              \
    } while (false);

#define GL_CHECK                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        GLenum gl_error;                                                                                               \
        int __count = 0;                                                                                                 \
        while ((gl_error = glGetError()) != GL_NO_ERROR)                                                               \
        {                                                                                                              \
            LOG_ERROR("GL error {} before line {} in file {}", GetGLErrorStr(gl_error).c_str(), __LINE__, __FILE__);           \
            if (++__count > 10)                                                                                          \
                break;                                                                                                 \
        }                                                                                                              \
    } while (0);

#endif

    struct texture1d_unit_t { GLint unit; };

    struct texture2d_unit_t { GLint unit; };

    struct texture3d_unit_t { GLint unit; };


    class gl_object_base_t: public no_copy_t
    {
    protected:
        GLuint handle_;
    public:

        explicit gl_object_base_t(GLuint handle = 0) noexcept
                :handle_(handle)
        {}

        gl_object_base_t(gl_object_base_t&& other) noexcept
                :handle_(other.handle_)
        {
            other.handle_ = 0;
        }

        gl_object_base_t& operator=(gl_object_base_t&& other) noexcept{
            assert(!handle_);
            handle_ = other.handle_;
            other.handle_ = 0;
            return *this;
        }

        ~gl_object_base_t(){
            assert(!handle_);
        }
        GLuint handle() const noexcept{
            return handle_;
        }
    };

    template <typename T> struct var_to_gl_type;

    template <>
    struct var_to_gl_type<float>{
        static constexpr GLenum type = GL_FLOAT;
        static constexpr GLenum stype = GL_FLOAT;
        static constexpr GLint  ncomp = 1;
    };

    template<>
    struct var_to_gl_type<int>{
        static constexpr GLenum type = GL_INT;
        static constexpr GLenum stype = GL_INT;
        static constexpr GLint  ncomp = 1;
    };

    template<>
    struct var_to_gl_type<unsigned int>{
        static constexpr GLenum type = GL_UNSIGNED_INT;
        static constexpr GLenum stype = GL_UNSIGNED_INT;
        static constexpr GLint  ncomp = 1;
    };

    template<>
    struct var_to_gl_type<vec2>{
        static constexpr GLenum type = GL_FLOAT_VEC2;
        static constexpr GLenum stype = GL_FLOAT;
        static constexpr GLint  ncomp = 2;
    };
    template<>
    struct var_to_gl_type<vec3>{
        static constexpr GLenum type = GL_FLOAT_VEC3;
        static constexpr GLenum stype = GL_FLOAT;
        static constexpr GLint  ncomp = 3;
    };
    template<>
    struct var_to_gl_type<vec4>{
        static constexpr GLenum type = GL_FLOAT_VEC4;
        static constexpr GLenum stype = GL_FLOAT;
        static constexpr GLint  ncomp = 4;
    };
    template<>
    struct var_to_gl_type<vec2i>{
        static constexpr GLenum type = GL_INT_VEC2;
        static constexpr GLenum stype = GL_INT;
        static constexpr GLint  ncomp = 2;
    };
    template<>
    struct var_to_gl_type<vec3i>{
        static constexpr GLenum type = GL_INT_VEC3;
        static constexpr GLenum stype = GL_INT;
        static constexpr GLint  ncomp = 3;
    };
    template<>
    struct var_to_gl_type<vec4i>{
        static constexpr GLenum type = GL_INT_VEC4;
        static constexpr GLenum stype = GL_INT;
        static constexpr GLint  ncomp = 4;
    };

    template<>
    struct var_to_gl_type<mat4>{
        static constexpr GLenum type = GL_FLOAT_MAT4;
        static constexpr GLenum stype = GL_FLOAT;
    };

    template<>
    struct var_to_gl_type<texture1d_unit_t>{
        static constexpr GLenum type = GL_SAMPLER_1D;
    };

    template<>
    struct var_to_gl_type<texture2d_unit_t>{
        static constexpr GLenum type = GL_SAMPLER_2D;
    };

    template<>
    struct var_to_gl_type<texture3d_unit_t>{
        static constexpr GLenum type = GL_SAMPLER_3D;
    };


VUTIL_GL_END

