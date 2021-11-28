/*
* SphereShapeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <Geometry/Shapes/SphereShape.h>

TEST(SphereShape, BasicOperations)
{
    SphereShape sphere(Vec4_0, SimdFloat_1);
    EXPECT_EQ(sphere.getRadius().getFloat(), 1.0f);
    EXPECT_TRUE(sphere.getCenter().exactEquals<3>(Vec4_0));

    sphere.setRadius(SimdFloat(2.5f));
    EXPECT_EQ(sphere.getRadius().getFloat(), 2.5f);

    sphere.setCenter(Vec4_1);
    EXPECT_TRUE(sphere.getCenter().exactEquals<3>(Vec4_1));
}

TEST(SphereShape, Queries)
{
    //
    // Ray cast
    //

    // The ray hits the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 start = Vec4_0;
        Vector4 end(4.f, 5.f, 6.f);
        Shape::RayCastOutput output;
        sphere.castRay(start, end, output);

        // Ray should hit
        EXPECT_TRUE(output.m_hit);
        EXPECT_TRUE(output.m_fraction > 0.f && output.m_fraction < 1.f);
        EXPECT_TRUE(output.m_hitNormal.isNormalized<3>());
        // Hit point should be on the sphere surface
        float x = (output.m_hitPoint.getComponent<0>() - rad).getFloat();
        float y = (output.m_hitPoint.getComponent<1>() - rad).getFloat();
        float z = (output.m_hitPoint.getComponent<2>() - rad).getFloat();
        EXPECT_TRUE(std::fabsf(x*x + y*y + z*z - (rad * rad).getFloat()) < 1E-5f);
        // Fraction should be correct.
        Vector4 p = start + (end - start) * SimdFloat(output.m_fraction);
        Vector4 diff = output.m_hitPoint - p;
        EXPECT_TRUE(diff.lengthSq<3>().getFloat() < 1E-5f);
    }

    // The ray misses the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Shape::RayCastOutput output;

        // Ray is going the opposite direction
        Vector4 start1 = Vec4_0;
        Vector4 end1(-2.f, -3.f, -4.f);
        sphere.castRay(start1, end1, output);
        EXPECT_FALSE(output.m_hit);

        // Ray is too short to reach to the sphere
        Vector4 start2(-2.f, -3.f, -4.f);
        Vector4 end2 = Vec4_0;
        sphere.castRay(start2, end2, output);
        EXPECT_FALSE(output.m_hit);

        // Ray is going toward direction of the sphere but misses
        Vector4 start3 = Vec4_0;
        Vector4 end3(5.f, 5.f, -1.f);
        sphere.castRay(start3, end3, output);
        EXPECT_FALSE(output.m_hit);
    }

    // The ray starts from inner region of the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 start = Vec4_1;
        Vector4 end(4.f, 5.f, 6.f);
        Shape::RayCastOutput output;
        sphere.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_EQ(output.m_fraction, 0.0f);
        EXPECT_TRUE(output.m_hitPoint.exactEquals<3>(start));
        Vector4 dir = start - center;
        dir.normalize<3>();
        EXPECT_TRUE(output.m_hitNormal.equals<3>(dir, SimdFloat(1E-5f)));
    }

    // The ray starts on the sphere surface and is going in
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 start(2.0f, 2.0f, 0.0f);
        Vector4 end(4.f, 5.f, 6.f);
        Shape::RayCastOutput output;
        sphere.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_TRUE(std::fabs(output.m_fraction) < 1E-5f);
        EXPECT_TRUE(output.m_hitPoint.equals<3>(start, SimdFloat(1E-5f)));
        Vector4 dir = start - center;
        dir.normalize<3>();
        EXPECT_TRUE(output.m_hitNormal.equals<3>(dir, SimdFloat(1E-5f)));
    }

    // The ray starts on the sphere surface and is going away
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 start(2.0f, 2.0f, 0.0f);
        Vector4 end(4.f, -5.f, -6.f);
        Shape::RayCastOutput output;
        sphere.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_EQ(output.m_fraction, 0.f);
        EXPECT_TRUE(output.m_hitPoint.exactEquals<3>(start));
    }

    // The ray ends on the sphere surface
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 start(-1.f, -1.f, -1.f);
        Vector4 end(2.0f, 2.0f, 0.0f);
        Shape::RayCastOutput output;
        sphere.castRay(start, end, output);
        EXPECT_TRUE(output.m_hit);
        EXPECT_EQ(output.m_fraction, 1.f);
        EXPECT_TRUE(output.m_hitPoint.equals<3>(end, SimdFloat(1E-5f)));
    }

    // The ray touches the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 start = Vec4_0;
        Vector4 end(5.0f, 5.0f, 0.0f);
        Shape::RayCastOutput output;
        sphere.castRay(start, end, output);
        EXPECT_FALSE(output.m_hit);
    }

    //
    // Closest point
    //

    // The point is outside the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 p = Vec4_0;
        Shape::ClosestPointOutput output;
        sphere.getClosestPoint(p, output);
        // Closest point should be on the sphere surface
        float x = (output.m_closestPoint.getComponent<0>() - rad).getFloat();
        float y = (output.m_closestPoint.getComponent<1>() - rad).getFloat();
        float z = (output.m_closestPoint.getComponent<2>() - rad).getFloat();
        EXPECT_TRUE(std::fabsf(x* x + y * y + z * z - (rad * rad).getFloat()) < 1E-5f);
        Vector4 dir = p - output.m_closestPoint;
        dir.normalize<3>();
        EXPECT_TRUE(output.m_normal.equals<3>(dir, SimdFloat(1E-5f)));
    }

    // The point is inside the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 p = Vec4_1;
        Shape::ClosestPointOutput output;
        sphere.getClosestPoint(p, output);
        // Closest point should be on the sphere surface
        float x = (output.m_closestPoint.getComponent<0>() - rad).getFloat();
        float y = (output.m_closestPoint.getComponent<1>() - rad).getFloat();
        float z = (output.m_closestPoint.getComponent<2>() - rad).getFloat();
        EXPECT_TRUE(std::fabsf(x* x + y * y + z * z - (rad * rad).getFloat()) < 1E-5f);
        Vector4 dir = p - center;
        dir.normalize<3>();
        EXPECT_TRUE(output.m_normal.equals<3>(dir, SimdFloat(1E-5f)));
    }

    // The point is on the sphere
    {
        SimdFloat rad = SimdFloat_2;
        Vector4 center = SimdFloat_2 * Vec4_1;
        SphereShape sphere(center, rad);
        Vector4 p(2.0f, 2.0f, 0.0f);
        Shape::ClosestPointOutput output;
        sphere.getClosestPoint(p, output);
        EXPECT_TRUE(output.m_closestPoint.exactEquals<3>(p));
        Vector4 dir = p - center;
        dir.normalize<3>();
        EXPECT_TRUE(output.m_normal.equals<3>(dir, SimdFloat(1E-5f)));
    }
}
