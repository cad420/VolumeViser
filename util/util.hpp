#pragma once

// misc
#include "misc.hpp"


// memory
#include "memory/alloc.hpp"
#include "memory/memory_arena.hpp"
#include "memory/object_pool.hpp"

// math
#include "math.hpp"


// geometry
#include "geometry.hpp"

// image
#include "image.hpp"

// file
#include "file/raw_file_io.hpp"

// console
#include "console/cmdline.hpp"
#include "console/logger.hpp"
#include "console/progressbar.hpp"

// event
#include "event/keycode.hpp"

// parallel
#include "parallel/thread_id.hpp"
#include "parallel/thread_group.hpp"
#include "parallel/parallel_for.hpp"

// opengl
#ifdef UTIL_ENABLE_OPENGL
#include "opengl/attrib.hpp"
#include "opengl/buffer.hpp"
#include "opengl/camera.hpp"
#include "opengl/demo.hpp"
#include "opengl/framebuffer.hpp"
#include "opengl/keyboard.hpp"
#include "opengl/mouse.hpp"
#include "opengl/program.hpp"
#include "opengl/renderbuffer.hpp"
#include "opengl/sampler.hpp"
#include "opengl/shader.hpp"
#include "opengl/texture.hpp"
#include "opengl/uniform.hpp"
#include "opengl/vertex_array.hpp"
#include "opengl/window.hpp"
#endif