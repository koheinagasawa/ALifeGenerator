/*
* SimdFloatTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <Common/Math/Simd/SimdFloat.h>

TEST(SimdFloat, BasicOperations)
{
    SimdFloat v1(1.5f);
    EXPECT_EQ(v1.getFloat(), 1.5f);

    SimdFloat v2(v1);
    EXPECT_EQ(v2.getFloat(), 1.5f);

    SimdFloat v3 = v1 + v2;
    EXPECT_EQ(v3.getFloat(), 3.0f);

    SimdFloat v4 = v3 * v1;
    EXPECT_EQ(v4.getFloat(), 4.5f);

    SimdFloat v5 = v4 - v3;
    EXPECT_EQ(v5.getFloat(), 1.5f);

    EXPECT_EQ((-v5).getFloat(), -1.5f);

    SimdFloat v6 = v4 / v5;
    EXPECT_EQ(v6.getFloat(), 3.0f);

    EXPECT_TRUE(v1 == v2);
    EXPECT_TRUE(v1 == v5);
    EXPECT_TRUE(v1 != v3);
    EXPECT_TRUE(v1 < v3);
    EXPECT_TRUE(v1 <= v3);
    EXPECT_TRUE(v1 <= v2);
    EXPECT_TRUE(v4 > v3);
    EXPECT_TRUE(v4 >= v3);
    EXPECT_TRUE(v3 >= v6);

    EXPECT_TRUE(std::fabsf(v3.getSqrt().getFloat() - std::sqrt(3.0f)) < 1E-5f);
    EXPECT_TRUE(std::fabsf(v3.getInverse().getFloat() - 1.f/3.0f) < 1E-5f);
}
