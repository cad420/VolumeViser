#pragma once

#include "../common.hpp"
#include "misc/hash.hpp"
#include "vec.hpp"

#include <ostream>

VUTIL_BEGIN

    template <typename T>
    class tvec2{
    public:
        using self_t = tvec2<T>;

        T x,y;

        tvec2() noexcept;
        tvec2(T x, T y) noexcept;
        explicit tvec2( T val) noexcept;
        explicit tvec2(uninitialized_t) noexcept;
        tvec2(const self_t& v) noexcept;
        tvec2(const tvec<T,2>& v) noexcept;

        bool is_zero() const noexcept;

        auto length() const noexcept;

        auto length_square() const noexcept;

        void normalize() noexcept;

        self_t normalized() const noexcept;

        template <typename U>
        auto convert_to() const noexcept;

        T& operator[](int idx) noexcept;
        const T& operator[](int idx) const noexcept;

        self_t& operator+=(const self_t& rhs) noexcept;
        self_t& operator-=(const self_t& rhs) noexcept;
        self_t& operator*=(const self_t& rhs) noexcept;
        self_t& operator/=(const self_t& rhs) noexcept;

        self_t& operator+=(T rhs) noexcept;
        self_t& operator-=(T rhs) noexcept;
        self_t& operator*=(T rhs) noexcept;
        self_t& operator/=(T rhs) noexcept;

#define SWIZZLE2(A, B)       tvec2<T> A##B()       const { return tvec2<T>(A, B); }
        SWIZZLE2(x, x) SWIZZLE2(x, y) SWIZZLE2(y, y) SWIZZLE2(y, x)
#undef SWIZZLE2

    };

    template <typename T>
    auto operator-(const tvec2<T>& vec) noexcept;

    template <typename T>
    auto operator+(const tvec2<T>& lhs,const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto operator-(const tvec2<T>& lhs,const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto operator*(const tvec2<T>& lhs,const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto operator/(const tvec2<T>& lhs,const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto operator+(const tvec2<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator-(const tvec2<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator*(const tvec2<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator/(const tvec2<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator+(T lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto operator*(T lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    bool operator==(const tvec2<T>& lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    bool operator!=(const tvec2<T>& lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    bool operator<(const tvec2<T>& lhs,const tvec2<T>& rhs) noexcept;

    template <typename T>
    bool operator>(const tvec2<T>& lhs,const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto dot(const tvec2<T>& lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto cross(const tvec2<T>& lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    auto cos(const tvec2<T>& lhs, const tvec2<T>& rhs) noexcept;

    template <typename T>
    std::ostream& operator<<(std::ostream& out, const tvec2<T>& vec);

    using vec2f = tvec2<float>;
    using vec2d = tvec2<double>;
    using vec2i = tvec2<int>;
    using vec2b = tvec2<unsigned char>;

VUTIL_END

namespace std{

    template <typename T>
    struct hash<vutil::tvec2<T>>{
    size_t operator()(const vutil::tvec2<T>& vec) const noexcept{
        return vutil::hash(vec.x,vec.y);
    }
};

}
