/*
* PlaneShape.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Geometry/Geometry.h>
#include <Geometry/Shapes/PlaneShape.h>

PlaneShape::PlaneShape(const Vector4& plane)
{
    setPlane(plane);
}

void PlaneShape::setPlane(const Vector4& plane)
{
    m_plane = plane;
    SimdFloat len = m_plane.length<3>();

    // Normalize the plane
    if (len > SimdFloat_0)
    {
        m_plane /= len;
        assert(m_plane.isNormalized<3>());
    }
    else
    {
        m_plane.setZero();
    }
}

void PlaneShape::castRay(const Vector4& startIn, const Vector4& endIn, RayCastOutput& out) const
{
    Vector4 start = startIn;
    start.setComponent<3>(SimdFloat_1);
    Vector4 end = endIn;
    end.setComponent<3>(SimdFloat_1);
    const SimdFloat a = start.dot<4>(m_plane);

    // We handle hits only from front face.
    if (a >= SimdFloat_0)
    {
        const SimdFloat b = end.dot<4>(m_plane);

        // Make sure that start and end are located in the opposite side of the plane
        if (a * b <= SimdFloat_0)
        {
            const Vector4 ray = end - start;
            const SimdFloat dot = ray.dot<3>(m_plane);

            if (dot != SimdFloat_0)
            {
                // Hit
                const SimdFloat fraction = -a / dot;

                out.m_hit = true;
                out.m_fraction = fraction.getFloat();
                out.m_hitPoint = start + ray * fraction;
                out.m_hitNormal = m_plane;
                return;
            }
        }
    }

    out.m_hit = false;
}

void PlaneShape::getClosestPoint(const Vector4& position, ClosestPointOutput& out) const
{
    Vector4 posW1 = position;
    posW1.setComponent<3>(SimdFloat_1);

    SimdFloat d = posW1.dot<4>(m_plane);

    out.m_closestPoint = position - d * m_plane;
    out.m_normal = m_plane;
}
