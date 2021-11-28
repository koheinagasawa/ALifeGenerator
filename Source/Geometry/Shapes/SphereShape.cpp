/*
* SphereShape.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Geometry/Geometry.h>
#include <Geometry/Shapes/SphereShape.h>

#include <cmath>

SphereShape::SphereShape(const Vector4& center, float radius)
    : SphereShape(center, SimdFloat(radius))
{
}

SphereShape::SphereShape(const Vector4& center, const SimdFloat& radius)
{
    assert(radius > SimdFloat_0);
    setCenter(center);
    setRadius(radius);
}

void SphereShape::castRay(const Vector4& start, const Vector4& end, RayCastOutput& out) const
{
    const Vector4 startToCenter = m_centerAndRadius - start;
    const SimdFloat lenStartToCenterSq = startToCenter.lengthSq<3>();
    const SimdFloat rad = m_centerAndRadius.getComponent<3>();
    const SimdFloat radSq(rad * rad);

    // Check if the start is inside the sphere
    if (lenStartToCenterSq <= radSq)
    {
        out.m_hit = true;
        out.m_fraction = 0.f;
        out.m_hitPoint = start;
        out.m_hitNormal = -startToCenter;
        out.m_hitNormal.normalize<3>();
        return;
    }

    const Vector4 ray = end - start;
    const SimdFloat rayLength = ray.length<3>();
    Vector4 dir = ray;
    dir.normalize<3>();
    const SimdFloat t = dir.dot<3>(startToCenter);

    // Return no hit if the ray is going away from the sphere
    if (t < SimdFloat_0)
    {
        out.m_hit = false;
        return;
    }

    const Vector4 startToPerp = dir * t;
    const Vector4 perp = startToCenter - startToPerp;
    const SimdFloat perpLenSq = perp.lengthSq<3>();

    if (perpLenSq < radSq)
    {
        // Hit
        const SimdFloat dist(startToPerp.length<3>().getFloat() - std::sqrtf((radSq - perpLenSq).getFloat()));
        float fraction = (dist / rayLength).getFloat();

        if (fraction <= 1.0f)
        {
            out.m_hit = true;
            out.m_fraction = fraction;
            out.m_hitPoint = start + dist * dir;
            out.m_hitNormal = out.m_hitPoint - m_centerAndRadius;
            out.m_hitNormal.normalize<3>();
            return;
        }
    }

    // The ray missed
    out.m_hit = false;
    return;
}

void SphereShape::getClosestPoint(const Vector4& position, ClosestPointOutput& out) const
{
    Vector4 dir = position - m_centerAndRadius;
    dir.normalize<3>();
    out.m_closestPoint = m_centerAndRadius + m_centerAndRadius.getComponent<3>() * dir;
    out.m_normal = dir;
    return;
}
