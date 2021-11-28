/*
* Shape.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/Math/Vector4.h>

// Abstract class of shape of geometry.
class Shape
{
public:

    // Return true if the shape has differentiation of interior and exterior.
    virtual bool hasInterior() const = 0;

    // Output struct of ray cast query.
    struct RayCastOutput
    {
        Vector4 m_hitPoint;     // The first hit point.
        Vector4 m_hitNormal;    // The hit normal.
        float m_fraction;       // Fraction from the start to the hit point. It has to be from 0 to 1.
        bool m_hit;             // True if there is a hit.
    };

    // Cast a ray from start to end and return an output.
    // Hitting from back face or inside the shape is not detected.
    virtual void castRay(const Vector4& start, const Vector4& end, RayCastOutput& out) const = 0;

    // Output struct of closest point query.
    struct ClosestPointOutput
    {
        Vector4 m_closestPoint; // The closest point.
        Vector4 m_normal;       // Face normal at the closest point.
    };

    // Find the closest point and its normal on the surface of this shape from the given position.
    // If the position is inside the shape, it returns the closest point to get out from the shape.
    virtual void getClosestPoint(const Vector4& position, ClosestPointOutput& out) const = 0;
};
