/*
* Vector4.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/Math/Simd/SimdFloat.h>

// 4-element vector type using float
class Vector4
{
public:
    // Constructors
    inline Vector4() = default;
    inline Vector4(const Vector4& v) : m_quad(v.m_quad) {}
    inline Vector4(Vector4&& v) : m_quad(v.m_quad) {}
    inline Vector4(float a, float b, float c, float d = float(0)) { set(a, b, c, d); }
    inline Vector4(const SimdFloat& a, const SimdFloat& b, const SimdFloat& c, const SimdFloat& d = SimdFloat_0) { set(a, b, c, d); }

    // Copy and move
    inline Vector4& operator= (const Vector4& v) { m_quad = v.m_quad; return *this; }
    inline Vector4& operator= (Vector4&& v) { m_quad = v.m_quad; return *this; }

    // Operators
    inline void operator+=(const Vector4& rhs);
    inline void operator-=(const Vector4& rhs);
    inline void operator*=(const Vector4& rhs); // Multiply each component
    inline void operator*=(const SimdFloat& rhs);
    inline void operator/=(const Vector4& rhs); // Divide each component
    inline void operator/=(const SimdFloat& rhs);
    inline Vector4 operator+(const Vector4& v) const { Vector4 out(*this); out += v; return out; }
    inline Vector4 operator-(const Vector4& v) const { Vector4 out(*this); out -= v; return out; }
    inline Vector4 operator*(const Vector4& v) const { Vector4 out(*this); out *= v; return out; } // Multiply each component
    inline Vector4 operator*(const SimdFloat& f) const { Vector4 out(*this); out *= f; return out; }
    inline Vector4 operator/(const Vector4& v) const { Vector4 out(*this); out /= v; return out; } // Divide each component
    inline Vector4 operator/(const SimdFloat& f) const { Vector4 out(*this); out /= f; return out; }
    inline Vector4 operator-() const { Vector4 out; out.setNegate(*this); return out; }

    // Return true if the first N components are equal to rhs within the error of eps.
    template<int N> inline bool equals(const Vector4& rhs, const SimdFloat& eps = SimdFloat(1E-4f)) const;
    // Return true if the first N components are exactly equal to rhs.
    template<int N> inline bool exactEquals(const Vector4& rhs) const;

    // Accessors to components
    constexpr float operator()(unsigned int index) const;
    constexpr float& operator()(unsigned int index);
    template<int> constexpr SimdFloat getComponent() const;
    template<int> constexpr void setComponent(const SimdFloat& v);

    // Setters
    inline void set(float a, float b, float c, float d = float(0));
    inline void set(const SimdFloat& a, const SimdFloat& b, const SimdFloat& c, const SimdFloat& d = SimdFloat_0);
    inline void setAll(float a);
    inline void setAll(const SimdFloat& a);
    inline void setZero(); // Set all component zero.

    // Set negative vector of the given v.
    inline void setNegate(const Vector4& v);

    // Set absolute vector of the given v.
    inline void setAbs(const Vector4& v);

    // Return dot product.
    template<int> inline SimdFloat dot(const Vector4& v) const;

    // Return cross product.
    inline static Vector4 cross(const Vector4& v1, const Vector4& v2);

    // Normalize the first N components of this vector.
    template<int N> inline void normalize() { this->operator*=(length<N>().getInverse()); }

    // Return true if the first N components of this vector is normalized within threshold error.
    template<int N> inline bool isNormalized(const SimdFloat& threshold = SimdFloat(1E-5f)) const { return (lengthSq<N>() - SimdFloat_1) < threshold; }

    // Return length of this vector using the first N components.
    template<int N> constexpr SimdFloat length() const { return lengthSq<N>().getSqrt(); }
    // Return length^2 of this vector using the first N components.
    template<int N> constexpr SimdFloat lengthSq() const { return dot<N>(*this); }

private:

    ALIGN16(QuadrupleFloat) m_quad; // The data.
};

static inline Vector4 operator*(const SimdFloat& f, const Vector4& v) { return v * f; }

// Static constant values
static const Vector4 Vec4_0 = Vector4(0.f, 0.f, 0.f, 0.f);
static const Vector4 Vec4_1 = Vector4(1.f, 1.f, 1.f, 1.f);
static const Vector4 Vec4_1000 = Vector4(1.f, 0.f, 0.f, 0.f);
static const Vector4 Vec4_0100 = Vector4(0.f, 1.f, 0.f, 0.f);
static const Vector4 Vec4_0010 = Vector4(0.f, 0.f, 1.f, 0.f);
static const Vector4 Vec4_0001 = Vector4(0.f, 0.f, 0.f, 1.f);

#ifdef USE_SSE

//
// SSE implementation
//

inline void Vector4::operator+=(const Vector4& rhs)
{
    m_quad = _mm_add_ps(m_quad, rhs.m_quad);
}

inline void Vector4::operator-=(const Vector4& rhs)
{
    m_quad = _mm_sub_ps(m_quad, rhs.m_quad);
}

inline void Vector4::operator*=(const Vector4& rhs)
{
    m_quad = _mm_mul_ps(m_quad, rhs.m_quad);
}

inline void Vector4::operator*=(const SimdFloat& rhs)
{
    m_quad = _mm_mul_ps(rhs.get(), m_quad);
}

inline void Vector4::operator/=(const SimdFloat& rhs)
{
    m_quad = _mm_mul_ps(rhs.getInverse().get(), m_quad);
}

inline void Vector4::operator/=(const Vector4& rhs)
{
    m_quad = _mm_div_ps(m_quad, rhs.m_quad);
}

template<>
inline bool Vector4::equals<3>(const Vector4& rhs, const SimdFloat& eps) const
{
    Vector4 sub = *this - rhs;
    sub.setAbs(sub);
    Vector4 epsV;
    epsV.setAll(eps);
    return (_mm_movemask_ps(_mm_cmple_ps(sub.m_quad, epsV.m_quad)) & 0b0111) == 0b0111;
}

template<>
inline bool Vector4::equals<4>(const Vector4& rhs, const SimdFloat& eps) const
{
    Vector4 sub = *this - rhs;
    sub.setAbs(sub);
    Vector4 epsV;
    epsV.setAll(eps);
    return _mm_movemask_ps(_mm_cmple_ps(sub.m_quad, epsV.m_quad)) == 0b1111;
}

template<> inline bool Vector4::exactEquals<3>(const Vector4& rhs) const
{
    return (_mm_movemask_ps(_mm_cmpeq_ps(m_quad, rhs.m_quad)) & 0b0111) == 0b0111;
}

template<> inline bool Vector4::exactEquals<4>(const Vector4& rhs) const
{
    return _mm_movemask_ps(_mm_cmpeq_ps(m_quad, rhs.m_quad)) == 0b1111;
}

inline constexpr float Vector4::operator()(unsigned int index) const
{
    return m_quad.m128_f32[index];
}

inline constexpr float& Vector4::operator()(unsigned int index)
{
    return m_quad.m128_f32[index];
}

template<int N>
constexpr SimdFloat Vector4::getComponent() const
{
    return SimdFloat(_mm_shuffle_ps(m_quad, m_quad, _MM_SHUFFLE(N, N, N, N)));
}

template<int N> constexpr void Vector4::setComponent(const SimdFloat& v)
{
    m_quad = _mm_blend_ps(m_quad, v.get(), 0x1 << N);
}

inline void Vector4::set(float a, float b, float c, float d)
{
    m_quad = _mm_setr_ps(a, b, c, d);
}

inline void Vector4::set(const SimdFloat& a, const SimdFloat& b, const SimdFloat& c, const SimdFloat& d)
{
    const QuadrupleFloat ab = _mm_unpacklo_ps(a.get(), b.get());
    const QuadrupleFloat cd = _mm_unpacklo_ps(c.get(), d.get());
    m_quad = _mm_movelh_ps(ab, cd);
}

inline void Vector4::setAll(float a)
{
    m_quad = _mm_set1_ps(a);
}

inline void Vector4::setAll(const SimdFloat& a)
{
    m_quad = a.get();
}

inline void Vector4::setZero()
{
    m_quad = _mm_setzero_ps();
}

inline void Vector4::setNegate(const Vector4& v)
{
    m_quad = _mm_sub_ps(_mm_set1_ps(0.0f), v.m_quad);
}

inline void Vector4::setAbs(const Vector4& v)
{
    m_quad = _mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80000000)), v.m_quad);
}

template<>
inline SimdFloat Vector4::dot<3>(const Vector4& v) const
{
    return SimdFloat(_mm_dp_ps(m_quad, v.m_quad, 0x7F));
}

template<>
inline SimdFloat Vector4::dot<4>(const Vector4& v) const
{
    return SimdFloat(_mm_dp_ps(m_quad, v.m_quad, 0xFF));
}

inline Vector4 Vector4::cross(const Vector4& v1, const Vector4& v2)
{
    Vector4 out;
    const QuadrupleFloat cross0 = _mm_mul_ps(v1.m_quad, _mm_shuffle_ps(v2.m_quad, v2.m_quad, _MM_SHUFFLE(3, 0, 2, 1)));
    const QuadrupleFloat cross1 = _mm_mul_ps(v2.m_quad, _mm_shuffle_ps(v1.m_quad, v1.m_quad, _MM_SHUFFLE(3, 0, 2, 1)));
    const QuadrupleFloat diff = _mm_sub_ps(cross0, cross1);
    out.m_quad = _mm_shuffle_ps(diff, diff, _MM_SHUFFLE(3, 0, 2, 1));
    return out;
}

#else

//
// Non SSE implementation
//

inline void Vector4::operator+=(const Vector4& rhs)
{
    m_quad.m_floats[0] += rhs.m_quad.m_floats[0];
    m_quad.m_floats[1] += rhs.m_quad.m_floats[1];
    m_quad.m_floats[2] += rhs.m_quad.m_floats[2];
    m_quad.m_floats[3] += rhs.m_quad.m_floats[3];
}

inline void Vector4::operator-=(const Vector4& rhs)
{
    m_quad.m_floats[0] -= rhs.m_quad.m_floats[0];
    m_quad.m_floats[1] -= rhs.m_quad.m_floats[1];
    m_quad.m_floats[2] -= rhs.m_quad.m_floats[2];
    m_quad.m_floats[3] -= rhs.m_quad.m_floats[3];
}

inline void Vector4::operator*=(const Vector4& rhs)
{
    m_quad.m_floats[0] *= rhs.m_quad.m_floats[0];
    m_quad.m_floats[1] *= rhs.m_quad.m_floats[1];
    m_quad.m_floats[2] *= rhs.m_quad.m_floats[2];
    m_quad.m_floats[3] *= rhs.m_quad.m_floats[3];
}

inline void Vector4::operator*=(const SimdFloat& rhs)
{
    m_quad.m_floats[0] *= rhs;
    m_quad.m_floats[1] *= rhs;
    m_quad.m_floats[2] *= rhs;
    m_quad.m_floats[3] *= rhs;
}

inline void Vector4::operator/=(const Vector4& rhs)
{
    m_quad.m_floats[0] /= rhs.m_quad.m_floats[0];
    m_quad.m_floats[1] /= rhs.m_quad.m_floats[1];
    m_quad.m_floats[2] /= rhs.m_quad.m_floats[2];
    m_quad.m_floats[3] /= rhs.m_quad.m_floats[3];
}

inline void Vector4::operator/=(const SimdFloat& rhs)
{
    const SimdFloat inv = SimdFloat_1 / rhs;
    *this *= inv;
}

template<>
inline bool Vector4::equals<3>(const Vector4& rhs, const SimdFloat& eps) const
{
    Vector4 sub = *this - rhs;
    sub.setAbs(sub);
    return sub.getComponent<0>() <= eps && sub.getComponent<1>() <= eps && sub.getComponent<2>() <= eps;
}

template<>
inline bool Vector4::equals<4>(const Vector4& rhs, const SimdFloat& eps) const
{
    Vector4 sub = *this - rhs;
    sub.setAbs(sub);
    return sub.getComponent<0>() <= eps && sub.getComponent<1>() <= eps && sub.getComponent<2>() <= eps && sub.getComponent<3>() <= eps;
}

template<> inline bool Vector4::exactEquals<3>(const Vector4& rhs) const
{
    return equals<3>(rhs, SimdFloat_0);
}

template<> inline bool Vector4::exactEquals<4>(const Vector4& rhs) const
{
    return equals<4>(rhs, SimdFloat_0);
}

inline constexpr float Vector4::operator()(unsigned int index) const
{
    return m_quad.m_floats[index];
}

inline constexpr float& Vector4::operator()(unsigned int index)
{
    return m_quad.m_floats[index];
}

template<int N>
constexpr SimdFloat Vector4::getComponent() const
{
    return m_quad.m_floats[N];
}

template<int N> constexpr void Vector4::setComponent(const SimdFloat& v)
{
    m_quad.m_floats[N] = v;
}

inline void Vector4::set(float a, float b, float c, float d)
{
    m_quad.m_floats[0] = a;
    m_quad.m_floats[1] = b;
    m_quad.m_floats[2] = c;
    m_quad.m_floats[3] = d;
}

inline void Vector4::set(const SimdFloat& a, const SimdFloat& b, const SimdFloat& c, const SimdFloat& d)
{
    m_quad.m_floats[0] = a.getFloat();
    m_quad.m_floats[1] = b.getFloat();
    m_quad.m_floats[2] = c.getFloat();
    m_quad.m_floats[3] = d.getFloat();
}

inline void Vector4::setAll(float a)
{
    set(a, a, a, a);
}

inline void Vector4::setAll(const SimdFloat& a)
{
    set(a, a, a, a);
}

inline void Vector4::setZero()
{
    this = Vec4_0;
}

inline void Vector4::setNegate(const Vector4& v)
{
    (*this) *= -1.f;
}

inline void Vector4::setAbs(const Vector4& v)
{
    m_quad.m_floats[0] = fabsf(m_quad.m_floats[0]);
    m_quad.m_floats[1] = fabsf(m_quad.m_floats[1]);
    m_quad.m_floats[2] = fabsf(m_quad.m_floats[2]);
    m_quad.m_floats[3] = fabsf(m_quad.m_floats[3]);
}

template<>
inline SimdFloat Vector4::dot<3>(const Vector4& v) const
{
    return m_quad.m_floats[0] * v.m_quad.m_floats[0] + m_quad.m_floats[1] * v.m_quad.m_floats[1] + m_quad.m_floats[2] * v.m_quad.m_floats[2];
}

template<>
inline SimdFloat Vector4::dot<4>(const Vector4& v) const
{
    return m_quad.m_floats[0] * v.m_quad.m_floats[0] + m_quad.m_floats[1] * v.m_quad.m_floats[1] + m_quad.m_floats[2] * v.m_quad.m_floats[2] + m_quad.m_floats[3] * v.m_quad.m_floats[3];
}

inline Vector4 Vector4::cross(const Vector4& v1, const Vector4& v2)
{
    return Vector4(v1.getComponent<1>()*v2.getComponent<2>() - v1.getComponent<2>() * v2.getComponent<1>(), v1.getComponent<2>() * v2.getComponent<0>() - v1.getComponent<0>() * v2.getComponent<2>(), v1.getComponent<0>() * v2.getComponent<1>() - v1.getComponent<1>() * v2.getComponent<0>());
}

#endif