#pragma once

#include "../common.hpp"
#include "vec.hpp"
#include "vec2.hpp"

VUTIL_BEGIN


    template <typename T>
    class tvec3{
    public:
        using self_t = tvec3<T>;

        T x,y,z;

        tvec3() noexcept;
        tvec3(T x, T y, T z) noexcept;
        tvec3( T val) noexcept;
        tvec3(uninitialized_t) noexcept;
        tvec3(const self_t& v) noexcept;
        tvec3(const tvec<T,3>& v) noexcept;

        template<typename U>
        tvec3(const tvec3<U>& v) noexcept;

        bool is_zero() const noexcept;

        bool is_finite() const noexcept;

        bool is_nan() const noexcept;

        auto length() const noexcept;

        auto length_square() const noexcept;

        void normalize() noexcept;

        self_t normalized() const noexcept;

        template <typename F>
        auto map(F&& f) const noexcept;

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

#include "impl/swizzle_vec3.inl"
    };

    template <typename T>
    auto operator-(const tvec3<T>& vec) noexcept;

    template <typename T>
    auto operator+(const tvec3<T>& lhs,const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto operator-(const tvec3<T>& lhs,const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto operator*(const tvec3<T>& lhs,const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto operator/(const tvec3<T>& lhs,const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto operator+(const tvec3<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator-(const tvec3<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator*(const tvec3<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator/(const tvec3<T>& lhs, T rhs) noexcept;

    template <typename T>
    auto operator+(T lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto operator*(T lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    bool operator==(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    bool operator!=(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    bool operator<(const tvec3<T>& lhs,const tvec3<T>& rhs) noexcept;

    template <typename T>
    bool operator>(const tvec3<T>& lhs,const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto dot(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto abs_dot(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;


    template <typename T>
    auto cross(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto cos(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;

    template <typename T>
    auto abs_cos(const tvec3<T>& lhs, const tvec3<T>& rhs) noexcept;


    template <typename T>
    std::ostream& operator<<(std::ostream& out, const tvec3<T>& vec);

    using vec3f = tvec3<float>;
    using vec3d = tvec3<double>;
    using vec3i = tvec3<int>;
    using vec3b = tvec3<unsigned char>;

VUTIL_END
namespace std{

    template <typename T>
    struct hash<vutil::tvec3<T>>{
    size_t operator()(const vutil::tvec3<T>& vec) const noexcept{
        return vutil::hash(vec.x,vec.y,vec.z);
    }
};

}
