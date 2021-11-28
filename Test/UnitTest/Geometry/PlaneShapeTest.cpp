/*
* PlaneShapeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <Geometry/Shapes/PlaneShape.h>

TEST(PlaneShape, BasicOperations)
{
    PlaneShape plane1(Vector4(1.f, 2.f, 3.f, -1.f));
    EXPECT_TRUE(plane1.getPlane().isNormalized<3>());

    PlaneShape plane2(Vector4(3.f, 6.f, 9.f, -3.f));
    EXPECT_TRUE(plane2.getPlane().isNormalized<3>());

    EXPECT_TRUE(plane1.getPlane().equals<3>(plane2.getPlane()));

    PlaneShape plane3(Vector4(3.f, 6.f, 9.f, -5.f));
    EXPECT_TRUE(plane3.getPlane().isNormalized<3>());

    EXPECT_TRUE(plane1.getPlane().equals<3>(plane3.getPlane()));
    EXPECT_FALSE(plane1.getPlane().equals<4>(plane3.getPlane()));

    PlaneShape plane4(Vector4(2.f, 1.f, -3.f, 0.f));
    EXPECT_TRUE(plane4.getPlane().isNormalized<3>());

    EXPECT_FALSE(plane1.getPlane().equals<3>(plane4.getPlane()));

    plane4.setPlane(Vector4(3.f, 6.f, 9.f, -3.f));
    EXPECT_TRUE(plane1.getPlane().equals<3>(plane4.getPlane()));
}

TEST(PlaneShape, Queries)
{
    //
    // Ray cast
    //

    // The ray hits the plane
    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 start(8.f, 9.f, 0.f);
        Vector4 end(-5.f, -6.f, -7.f);
        Shape::RayCastOutput output;

        // Ray shouldn't hit back face.
        plane.castRay(end, start, output);
        EXPECT_FALSE(output.m_hit);

        // Ray should hit.
        plane.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_TRUE(output.m_fraction > 0.f && output.m_fraction < 1.f);
        EXPECT_TRUE(output.m_hitNormal.isNormalized<3>());
        // Hit point should be on the plane
        Vector4 pos = output.m_hitPoint;
        pos.setComponent<3>(SimdFloat_1);
        EXPECT_TRUE(std::fabsf(pos.dot<4>(plane.getPlane()).getFloat()) < 1E-5f);
        // Fraction should be correct.
        Vector4 p = start + (end - start) * SimdFloat(output.m_fraction);
        Vector4 diff = output.m_hitPoint - p;
        EXPECT_TRUE(diff.lengthSq<3>().getFloat() < 1E-5f);
    }

    // The ray misses the plane
    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 start(8.f, 9.f, 0.f);
        Vector4 end(5.f, 6.f, 7.f);
        Shape::RayCastOutput output;
        plane.castRay(start, end, output);
        EXPECT_FALSE(output.m_hit);
    }

    // The ray starts from the plane
    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 start(0.f, -2.f, 0.f);
        Vector4 end(-5.f, -6.f, -7.f);
        Shape::RayCastOutput output;
        plane.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_EQ(output.m_fraction, 0.0f);
        EXPECT_TRUE(output.m_hitPoint.exactEquals<3>(start));

        // Shift the start point slightly off the plane
        Vector4 start2(0.f, -1.999f, 0.f);
        plane.castRay(start2, end, output);
        EXPECT_TRUE(output.m_hit);
    }

    // The ray ends on the plane
    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 start(8.f, 9.f, 0.f);
        Vector4 end(0.f, -2.f, 0.f);
        Shape::RayCastOutput output;
        plane.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_EQ(output.m_fraction, 1.0f);
        EXPECT_TRUE(output.m_hitPoint.equals<3>(end, SimdFloat(1E-5f)));

        // Shift the end point slightly off the plane
        Vector4 end2(0.f, -2.0001f, 0.f);
        plane.castRay(start, end2, output);
        EXPECT_TRUE(output.m_hit);
    }

    // The ray is on the plane
    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 start(0.f, -2.f, 0.f);
        Vector4 end(2.f, 0.f, -2.f);
        Shape::RayCastOutput output;
        plane.castRay(start, end, output);
        EXPECT_FALSE(output.m_hit);

        // Shift the points slightly off the plane
        Vector4 start2(0.f, -1.999f, 0.f);
        Vector4 end2(1.9999f, 0.f, -2.0001f);
        plane.castRay(start2, end2, output);
        EXPECT_TRUE(output.m_hit);
    }

    //
    // Closest point
    //

    // The point is on the plane
    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 p(0.f, -2.f, 0.f);
        Shape::ClosestPointOutput output;
        plane.getClosestPoint(p, output);
        Vector4 pos = output.m_closestPoint;
        pos.setComponent<3>(SimdFloat_1);
        EXPECT_TRUE(std::fabsf(pos.dot<4>(plane.getPlane()).getFloat()) < 1E-5f);
        EXPECT_TRUE(p.equals<3>(output.m_closestPoint, SimdFloat(1E-5f)));
        EXPECT_TRUE(plane.getPlane().equals<3>(output.m_normal, SimdFloat(1E-5f)));
    }

    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 p(5.f, 6.f, 7.f);
        Shape::ClosestPointOutput output;
        plane.getClosestPoint(p, output);
        Vector4 pos = output.m_closestPoint;
        pos.setComponent<3>(SimdFloat_1);
        EXPECT_TRUE(std::fabsf(pos.dot<4>(plane.getPlane()).getFloat()) < 1E-5f);
        Vector4 d = p - output.m_closestPoint;
        d.normalize<3>();
        EXPECT_TRUE(d.equals<3>(plane.getPlane(), SimdFloat(1E-5f)));
        EXPECT_TRUE(plane.getPlane().equals<3>(output.m_normal, SimdFloat(1E-5f)));
    }

    {
        PlaneShape plane(Vector4(1.f, 2.f, 3.f, 4.f));
        Vector4 p(-8.f, -9.f, 0.f);
        Shape::ClosestPointOutput output;
        plane.getClosestPoint(p, output);
        Vector4 pos = output.m_closestPoint;
        pos.setComponent<3>(SimdFloat_1);
        EXPECT_TRUE(std::fabsf(pos.dot<4>(plane.getPlane()).getFloat()) < 1E-5f);
        Vector4 d = output.m_closestPoint - p;
        d.normalize<3>();
        EXPECT_TRUE(d.equals<3>(plane.getPlane(), SimdFloat(1E-5f)));
        EXPECT_TRUE(plane.getPlane().equals<3>(output.m_normal, SimdFloat(1E-5f)));
    }
}
