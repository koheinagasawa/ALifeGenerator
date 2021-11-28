/*
* PBDConstraints.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Solvers/PBD/PBDSolver.h>

namespace PBD
{
    // A constraint which tries to maintain the original length between two points.
    struct StretchConstraint : public Constraint
    {
        StretchConstraint(
        const SimdFloat& length, Vector4* positionA, Vector4* positionB,
        const SimdFloat& massA, const SimdFloat& massB, const SimdFloat& stiffness);

        virtual void project() override;

        Vector4* m_positionA;
        Vector4* m_positionB;
        SimdFloat m_massA, m_massB;
        SimdFloat m_length;
    };
}