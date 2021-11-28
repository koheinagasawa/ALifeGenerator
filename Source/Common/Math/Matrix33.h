/*
* Matrix33.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/Math/Vector4.h>

#include <cassert>
#include <memory.h>

// 3x3 matrix type using float
class Matrix33
{
public:
    // Constructors
    inline Matrix33() = default;
    inline Matrix33(const Vector4& col0, const Vector4& col1, const Vector4& col2);
    inline Matrix33(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22);
    inline Matrix33(const SimdFloat& m00, const SimdFloat& m01, const SimdFloat& m02, const SimdFloat& m10, const SimdFloat& m11, const SimdFloat& m12, const SimdFloat& m20, const SimdFloat& m21, const SimdFloat& m22);
    inline Matrix33(const Matrix33& m);
    inline Matrix33(Matrix33&& m);

    // Copy and move
    inline Matrix33& operator= (const Matrix33& m);
    inline Matrix33& operator= (Matrix33&& m);

    // Operators
    inline void operator+=(const Matrix33& rhs);
    inline void operator-=(const Matrix33& rhs);
    inline void operator*=(const Matrix33& rhs);
    inline void operator*=(const SimdFloat& f);
    inline Matrix33 operator+(const Matrix33& rhs) const { Matrix33 out(*this); out += rhs; return out; }
    inline Matrix33 operator-(const Matrix33& rhs) const { Matrix33 out(*this); out -= rhs; return out; }
    inline Matrix33 operator*(const Matrix33& rhs) const { Matrix33 out(*this); out *= rhs; return out; }
    inline Vector4 operator*(const Vector4& v) const;
    inline Matrix33 operator-() const { Matrix33 out; out.setNegate(*this); return out; }

    // Return true if the first N components are equal to rhs within the error of eps.
    inline bool equals(const Matrix33& rhs, const SimdFloat& eps = SimdFloat(1E-4f)) const;
    // Return true if the first N components are exactly equal to rhs.
    inline bool exactEquals(const Matrix33& rhs) const;

    // Accessors to components
    constexpr float operator()(int i, int j) const;
    constexpr float& operator()(int i, int j);
    template<int> constexpr const Vector4& getColumn() const;
    template<int> constexpr void setColumn(const Vector4& v);
    constexpr const Vector4& getColumn(int column) const;
    inline void setColumn(int column, const Vector4& v);
    template<int> inline Vector4 getRow() const;
    template<int> inline void setRow(const Vector4& v);
    inline Vector4 getRow(int column) const;
    inline void setRow(int column, const Vector4& v);
    template<int, int> constexpr SimdFloat getComponent() const;
    template<int, int> constexpr void setComponent(const SimdFloat& a);

    // Setters
    inline void setDiagonal(const SimdFloat& a);
    inline void setAll(const SimdFloat& a);
    inline void setZero();
    inline void setIdentity();

    // Set negative matrix of the given v.
    inline void setNegate(const Matrix33& m);

    // Set transpose of m to this matrix.
    inline void setTranspose(const Matrix33& m);

    // Return transpose of this matrix.
    inline Matrix33 transpose() const { Matrix33 out; out.setTranspose(*this); return out; }

    // Set inverse of m to this matrix.
    inline void setInverse(const Matrix33& m);

    // Return inverse of this matrix.
    inline Matrix33 inverse() const { Matrix33 out; out.setInverse(*this); return out; }

    // Return determinant of this matrix.
    inline SimdFloat getDeterminant() const;

protected:
    // The value
    Vector4 m_cols[3];
};

// Static constant values
static const Matrix33 Mat33_0 = Matrix33(Vec4_0, Vec4_0, Vec4_0);
static const Matrix33 Mat33_I = Matrix33(Vec4_1000, Vec4_0100, Vec4_0010);

// [TODO] Implement SIMD version

inline Matrix33::Matrix33(const Vector4& col0, const Vector4& col1, const Vector4& col2)
{
    m_cols[0] = col0;
    m_cols[1] = col1;
    m_cols[2] = col2;
}

inline Matrix33::Matrix33(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
{
    m_cols[0].set(m00, m01, m02);
    m_cols[1].set(m10, m11, m12);
    m_cols[2].set(m20, m21, m22);
}

inline Matrix33::Matrix33(const SimdFloat& m00, const SimdFloat& m01, const SimdFloat& m02, const SimdFloat& m10, const SimdFloat& m11, const SimdFloat& m12, const SimdFloat& m20, const SimdFloat& m21, const SimdFloat& m22)
{
    m_cols[0].set(m00, m01, m02);
    m_cols[1].set(m10, m11, m12);
    m_cols[2].set(m20, m21, m22);
}

inline Matrix33::Matrix33(const Matrix33& m)
{
    memcpy(m_cols, m.m_cols, sizeof(Matrix33));
}

inline Matrix33::Matrix33(Matrix33&& m)
{
    m_cols[0] = m.m_cols[0];
    m_cols[1] = m.m_cols[1];
    m_cols[2] = m.m_cols[2];
}

inline Matrix33& Matrix33::operator= (const Matrix33& m)
{
    memcpy(m_cols, m.m_cols, sizeof(Matrix33));
    return *this;
}

inline Matrix33& Matrix33::operator= (Matrix33&& m)
{
    m_cols[0] = m.m_cols[0];
    m_cols[1] = m.m_cols[1];
    m_cols[2] = m.m_cols[2];
    return *this;
}

inline void Matrix33::operator+=(const Matrix33& rhs)
{
    m_cols[0] += rhs.m_cols[0];
    m_cols[1] += rhs.m_cols[1];
    m_cols[2] += rhs.m_cols[2];
}

inline void Matrix33::operator-=(const Matrix33& rhs)
{
    m_cols[0] -= rhs.m_cols[0];
    m_cols[1] -= rhs.m_cols[1];
    m_cols[2] -= rhs.m_cols[2];
}

inline void Matrix33::operator*=(const Matrix33& rhs)
{
    Vector4 r0 = getRow<0>();
    Vector4 r1 = getRow<1>();
    Vector4 r2 = getRow<2>();

    m_cols[0].set(r0.dot<3>(rhs.getColumn<0>()), r1.dot<3>(rhs.getColumn<0>()), r2.dot<3>(rhs.getColumn<0>()));
    m_cols[1].set(r0.dot<3>(rhs.getColumn<1>()), r1.dot<3>(rhs.getColumn<1>()), r2.dot<3>(rhs.getColumn<1>()));
    m_cols[2].set(r0.dot<3>(rhs.getColumn<2>()), r1.dot<3>(rhs.getColumn<2>()), r2.dot<3>(rhs.getColumn<2>()));
}

inline void Matrix33::operator*=(const SimdFloat& f)
{
    m_cols[0] *= f;
    m_cols[1] *= f;
    m_cols[2] *= f;
}

inline Vector4 Matrix33::operator*(const Vector4& v) const
{
    return getColumn<0>() * v.getComponent<0>() + getColumn<1>() * v.getComponent<1>() + getColumn<2>() * v.getComponent<2>();
}

inline bool Matrix33::equals(const Matrix33& rhs, const SimdFloat& eps) const
{
    return m_cols[0].equals<3>(rhs.m_cols[0], eps) && m_cols[1].equals<3>(rhs.m_cols[1], eps) && m_cols[2].equals<3>(rhs.m_cols[2], eps);
}

inline bool Matrix33::exactEquals(const Matrix33& rhs) const
{
    return m_cols[0].exactEquals<3>(rhs.m_cols[0]) && m_cols[1].exactEquals<3>(rhs.m_cols[1]) && m_cols[2].exactEquals<3>(rhs.m_cols[2]);
}

constexpr float Matrix33::operator()(int i, int j) const
{
    return m_cols[j](i);
}

constexpr float& Matrix33::operator()(int i, int j)
{
    return m_cols[j](i);
}

template<int COL>
constexpr const Vector4& Matrix33::getColumn() const
{
    return m_cols[COL];
}

template<int COL>
constexpr void Matrix33::setColumn(const Vector4& v)
{
    m_cols[COL] = v;
}

constexpr const Vector4& Matrix33::getColumn(int column) const
{
    return m_cols[column];
}

inline void Matrix33::setColumn(int column, const Vector4& v)
{
    m_cols[column] = v;
}

template<int ROW>
inline Vector4 Matrix33::getRow() const
{
    return Vector4(m_cols[0].getComponent<ROW>(), m_cols[1].getComponent<ROW>(), m_cols[2].getComponent<ROW>(), SimdFloat_0);
}

template<int ROW>
inline void Matrix33::setRow(const Vector4& v)
{
    m_cols[0].setComponent<ROW>(v.getComponent<0>());
    m_cols[1].setComponent<ROW>(v.getComponent<1>());
    m_cols[2].setComponent<ROW>(v.getComponent<2>());
}

inline Vector4 Matrix33::getRow(int row) const
{
    switch (row)
    {
    case 0:
        return getRow<0>();
    case 1:
        return getRow<1>();
    case 2:
        return getRow<2>();
    default:
        assert(0);
        break;
    }
    return Vec4_0;
}

inline void Matrix33::setRow(int row, const Vector4& v)
{
    switch (row)
    {
    case 0:
        setRow<0>(v);
        break;
    case 1:
        setRow<1>(v);
        break;
    case 2:
        setRow<2>(v);
        break;
    default:
        assert(0);
        break;
    }
}

template<int ROW, int COL>
constexpr SimdFloat Matrix33::getComponent() const
{
    return m_cols[COL].getComponent<ROW>();
}

template<int ROW, int COL>
constexpr void Matrix33::setComponent(const SimdFloat& a)
{
    m_cols[COL].setComponent<ROW>(a);
}

inline void Matrix33::setDiagonal(const SimdFloat& a)
{
    setComponent<0, 0>(a);
    setComponent<1, 1>(a);
    setComponent<2, 2>(a);
}

inline void Matrix33::setAll(const SimdFloat& a)
{
    m_cols[0].setAll(a);
    m_cols[1].setAll(a);
    m_cols[2].setAll(a);
}

inline void Matrix33::setZero()
{
    *this = Mat33_0;
}

inline void Matrix33::setIdentity()
{
    *this = Mat33_I;
}

inline void Matrix33::setNegate(const Matrix33& m)
{
    m_cols[0] = -m.m_cols[0];
    m_cols[1] = -m.m_cols[1];
    m_cols[2] = -m.m_cols[2];
}

inline void Matrix33::setTranspose(const Matrix33& m)
{
    setRow<0>(m.getColumn<0>());
    setRow<1>(m.getColumn<1>());
    setRow<2>(m.getColumn<2>());
}

inline void Matrix33::setInverse(const Matrix33& m)
{
    SimdFloat det = m.getDeterminant();
    if (det == SimdFloat_0)
    {
        // [TODO] Better error handling.
        assert(0);
        return;
    }

    SimdFloat m00 = m.getColumn<0>().getComponent<0>();
    SimdFloat m01 = m.getColumn<1>().getComponent<0>();
    SimdFloat m02 = m.getColumn<2>().getComponent<0>();
    SimdFloat m10 = m.getColumn<0>().getComponent<1>();
    SimdFloat m11 = m.getColumn<1>().getComponent<1>();
    SimdFloat m12 = m.getColumn<2>().getComponent<1>();
    SimdFloat m20 = m.getColumn<0>().getComponent<2>();
    SimdFloat m21 = m.getColumn<1>().getComponent<2>();
    SimdFloat m22 = m.getColumn<2>().getComponent<2>();
    setColumn<0>(Vector4(m11 * m22 - m12 * m21, m21 * m02 - m01 * m22, m01 * m12 - m02 * m11));
    setColumn<1>(Vector4(m20 * m12 - m10 * m22, m00 * m22 - m20 * m02, m02 * m10 - m00 * m12));
    setColumn<2>(Vector4(m10 * m21 - m20 * m11, m01 * m20 - m00 * m21, m00 * m11 - m01 * m10));

    (*this) *= SimdFloat_1 / det;
}

inline SimdFloat Matrix33::getDeterminant() const
{
    SimdFloat m00 = getColumn<0>().getComponent<0>();
    SimdFloat m01 = getColumn<1>().getComponent<0>();
    SimdFloat m02 = getColumn<2>().getComponent<0>();
    SimdFloat m10 = getColumn<0>().getComponent<1>();
    SimdFloat m11 = getColumn<1>().getComponent<1>();
    SimdFloat m12 = getColumn<2>().getComponent<1>();
    SimdFloat m20 = getColumn<0>().getComponent<2>();
    SimdFloat m21 = getColumn<1>().getComponent<2>();
    SimdFloat m22 = getColumn<2>().getComponent<2>();
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
}
