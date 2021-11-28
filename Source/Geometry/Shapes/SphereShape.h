/*
* SphereShape.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Geometry/Shapes/Shape.h>
#include <Common/Math/Vector4.h>

// Sphere shape
class SphereShape : public Shape
{
public:
    //
    // Constructors
    //
    SphereShape(const Vector4& center, float radius);
    SphereShape(const Vector4& center, const SimdFloat& radius);

    //
    // Accessors to the center and radius.
    //
    inline void setRadius(float radius) { setRadius(SimdFloat(radius)); }
    inline void setRadius(const SimdFloat& radius) { assert(radius > SimdFloat_0); m_centerAndRadius.setComponent<3>(radius); }
    inline auto getRadius() const->SimdFloat { return m_centerAndRadius.getComponent<3>(); }
    inline void setCenter(const Vector4& center) { SimdFloat rad = m_centerAndRadius.getComponent<3>(); m_centerAndRadius = center; m_centerAndRadius.setComponent<3>(rad); }
    inline auto getCenter() const->const Vector4& { return m_centerAndRadius; }

    //
    // Query interface
    //
    virtual void castRay(const Vector4& start, const Vector4& end, RayCastOutput& out) const override;
    virtual void getClosestPoint(const Vector4& position, ClosestPointOutput& out) const override;

    // Sphere has interior, so return true.
    virtual bool hasInterior() const override { return true; }

    // Return true if the sphere is valid.
    inline bool isValid() const { return m_centerAndRadius.getComponent<3>() > SimdFloat_0; }

protected:
    Vector4 m_centerAndRadius;
};
