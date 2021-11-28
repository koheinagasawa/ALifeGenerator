/*
* PlaneShape.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Geometry/Shapes/Shape.h>
#include <Common/Math/Vector4.h>

// Plane shape
class PlaneShape : public Shape
{
public:
    //
    // Constructors
    //
    PlaneShape() = default;
    PlaneShape(const Vector4& plane);

    //
    // Accessors to the plane.
    //
    auto getPlane() const->const Vector4& { return m_plane; }
    void setPlane(const Vector4& plane);

    //
    // Query interface
    //
    virtual void castRay(const Vector4& start, const Vector4& end, RayCastOutput& out) const override;
    virtual void getClosestPoint(const Vector4& position, ClosestPointOutput& out) const override;

    // Plane has no interior, so return false.
    virtual bool hasInterior() const override { return false; }

    // Return true if the plane is valid.
    inline bool isValid() const { return m_plane.lengthSq<3>() > SimdFloat_0; }

protected:
    // The plane represented by equation Ax + By + Cz + D = 0.
    // This Vector4 stores (A, B, C, D) and they are normalized (|A, B, C| = 1).
    Vector4 m_plane = Vec4_0;
};
