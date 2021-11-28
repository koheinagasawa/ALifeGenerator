/*
* SimdFloat.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/Math/Simd/SseTypes.h>

// Wrapper class of float value represented by SIMD data type
class SimdFloat
{
public:
    // Constructors
    inline explicit SimdFloat(float val);
    inline SimdFloat(SingleFloat val) : m_val(val) {}
    inline SimdFloat(const SimdFloat& rhs) : m_val(rhs.m_val) {}
    inline SimdFloat(SimdFloat&& rhs) : m_val(rhs.m_val) {}

    // Operators
    inline void operator=(const SimdFloat& rhs) { m_val = rhs.m_val; }
    inline void operator=(SimdFloat&& rhs) { m_val = rhs.m_val; }
    inline SimdFloat operator+(const SimdFloat& rhs) const;
    inline SimdFloat operator-(const SimdFloat& rhs) const;
    inline SimdFloat operator*(const SimdFloat& rhs) const;
    inline SimdFloat operator/(const SimdFloat& rhs) const;
    inline SimdFloat operator-() const;
    inline void operator+=(const SimdFloat& rhs) { *this = *this + rhs; }
    inline void operator-=(const SimdFloat& rhs) { *this = *this - rhs; }
    inline void operator*=(const SimdFloat& rhs) { *this = *this * rhs; }
    inline void operator/=(const SimdFloat& rhs) { *this = *this / rhs; }
    inline bool operator==(const SimdFloat& rhs) const;
    inline bool operator!=(const SimdFloat& rhs) const { return !(*this == rhs); }
    inline bool operator<(const SimdFloat& rhs) const;
    inline bool operator<=(const SimdFloat& rhs) const;
    inline bool operator>(const SimdFloat& rhs) const;
    inline bool operator>=(const SimdFloat& rhs) const;

    // Return the value as float
    inline float getFloat() const;

    // Return square root of the value
    inline SimdFloat getSqrt() const;

    // Return inverse of the value
    inline SimdFloat getInverse() const;

    // Access to the underlying data
    inline const SingleFloat& get() const { return m_val; }

private:
    // The value
    SingleFloat m_val;
};

// Static constant values
const static SimdFloat SimdFloat_0(0.0f);
const static SimdFloat SimdFloat_1(1.0f);
const static SimdFloat SimdFloat_2(2.0f);

#ifdef USE_SSE

//
// SSE implementation
//

// Static constant values
extern const QuadrupleFloat QuadFloat1111;

inline SimdFloat::SimdFloat(float val)
{
    SingleFloat sf = _mm_load_ss(&val);
    m_val = _mm_shuffle_ps(sf, sf, _MM_SHUFFLE(0, 0, 0, 0));
}

inline SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const { return SimdFloat(_mm_add_ps(m_val, rhs.m_val)); }
inline SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const { return SimdFloat(_mm_sub_ps(m_val, rhs.m_val)); }
inline SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const { return SimdFloat(_mm_mul_ps(m_val, rhs.m_val)); }
inline SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const { return SimdFloat(_mm_div_ps(m_val, rhs.m_val)); }
inline SimdFloat SimdFloat::operator-() const { return SimdFloat(_mm_xor_ps(m_val, _mm_set1_ps(-0.0f))); }

inline bool SimdFloat::operator==(const SimdFloat& rhs) const { return _mm_ucomieq_ss(m_val, rhs.m_val); }
inline bool SimdFloat::operator<(const SimdFloat& rhs) const { return _mm_ucomilt_ss(m_val, rhs.m_val); }
inline bool SimdFloat::operator<=(const SimdFloat& rhs) const { return _mm_ucomile_ss(m_val, rhs.m_val); }
inline bool SimdFloat::operator>(const SimdFloat& rhs) const { return _mm_ucomigt_ss(m_val, rhs.m_val); }
inline bool SimdFloat::operator>=(const SimdFloat& rhs) const { return _mm_ucomige_ss(m_val, rhs.m_val); }

inline float SimdFloat::getFloat() const { return _mm_cvtss_f32(m_val); }
inline SimdFloat SimdFloat::getSqrt() const { return SimdFloat(_mm_sqrt_ps(m_val)); }
inline SimdFloat SimdFloat::getInverse() const { return SimdFloat(_mm_div_ps(QuadFloat1111, m_val)); }

#else

//
// Non SSE implementation
//

inline SimdFloat::SimdFloat(float val) : m_val(val) {}

inline SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const { return SimdFloat(m_val + rhs.m_val); }
inline SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const { return SimdFloat(m_val - rhs.m_val); }
inline SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const { return SimdFloat(m_val * rhs.m_val); }
inline SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const { return SimdFloat(m_val / rhs.m_val); }
inline SimdFloat SimdFloat::operator-() const { return SimdFloat(-m_val); }

inline bool SimdFloat::operator==(const SimdFloat& rhs) const { return m_val == rhs.m_val; }
inline bool SimdFloat::operator<(const SimdFloat& rhs) const { return m_val < rhs.m_val; }
inline bool SimdFloat::operator<=(const SimdFloat& rhs) const { return m_val <= rhs.m_val; }
inline bool SimdFloat::operator>(const SimdFloat& rhs) const { return m_val > rhs.m_val; }
inline bool SimdFloat::operator>=(const SimdFloat& rhs) const { return m_val >= rhs.m_val; }

inline float SimdFloat::getFloat() const { return m_val; }
inline SimdFloat SimdFloat::getSqrt() { return SimdFloat(sqrtf(m_val)); }
inline SimdFloat SimdFloat::getInverse() { return SimdFloat(1.f / m_val); }

#endif
