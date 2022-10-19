#pragma once

#include "../common.hpp"
#include "mat4_c.hpp"
#include "mat4_r.hpp"
#include "vec3.hpp"
#include "coord.hpp"

VUTIL_BEGIN

    template<typename T, bool COLUMN_MAJOR = true>
    class ttransform3
    {
        // TODO: row-major mat4
        using mat_t = std::conditional_t<COLUMN_MAJOR,
                tmat4_c<T>,tmat4_r<T>>;
        mat_t mat_;
        mat_t inv_;

    public:

        using self_t = ttransform3<T, COLUMN_MAJOR>;

        ttransform3()                                 noexcept;
        ttransform3(const mat_t &m, const mat_t &inv) noexcept;
        explicit ttransform3(const mat_t &m)          noexcept;
        explicit ttransform3(uninitialized_t)         noexcept;

        static self_t translate(const tvec3<T> &offset) noexcept;
        static self_t translate(T x, T y, T z)          noexcept;

        static self_t rotate(const tvec3<T> &axis, T rad) noexcept;
        static self_t rotate_x(T rad)                     noexcept;
        static self_t rotate_y(T rad)                     noexcept;
        static self_t rotate_z(T rad)                     noexcept;

        static self_t scale(const tvec3<T> &ratio) noexcept;
        static self_t scale(T x, T y, T z)         noexcept;

        self_t operator*=(const self_t &rhs) noexcept;

        tvec3<T>   apply_to_point (const tvec3<T> &point)   const noexcept;
        tvec3<T>   apply_to_vector(const tvec3<T> &vector)  const noexcept;
        tvec3<T>   apply_to_normal(const tvec3<T> &normal)  const noexcept;
        tcoord3<T> apply_to_coord (const tcoord3<T> &coord) const noexcept;

        tvec3<T>   apply_inverse_to_point (const tvec3<T> &point)   const noexcept;
        tvec3<T>   apply_inverse_to_vector(const tvec3<T> &vector)  const noexcept;
        tvec3<T>   apply_inverse_to_normal(const tvec3<T> &normal)  const noexcept;
        tcoord3<T> apply_inverse_to_coord (const tcoord3<T> &coord) const noexcept;

        self_t inv()     const noexcept;
        self_t inverse() const noexcept;

        const mat_t &get_mat()     const noexcept;
        const mat_t &get_inv_mat() const noexcept;
    };

    template<typename T, bool C>
    ttransform3<T, C> operator*(
            const ttransform3<T, C> &lhs, const ttransform3<T, C> &rhs) noexcept;

    using transform3f = ttransform3<float>;
    using transform3d = ttransform3<double>;


VUTIL_END