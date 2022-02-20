/*
* PBDConstraints.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Physics/Physics.h>
#include <Physics/Solvers/PBD/Constraints/PBDConstraints.h>

namespace PBD
{
    StretchConstraint::StretchConstraint(
        const SimdFloat& length, Vector4* positionA, Vector4* positionB,
        const SimdFloat& massA, const SimdFloat& massB, const SimdFloat& stiffness)
        : Constraint(stiffness)
        , m_positionA(positionA), m_positionB(positionB)
        , m_massA(massA), m_massB(massB)
        , m_length(length)
    {
    }

    void StretchConstraint::project()
    {
        Vector4 dir = *m_positionA - *m_positionB;
        const SimdFloat curLength = dir.length<3>();
        if (dir.lengthSq<3>().getFloat() > std::numeric_limits<float>::epsilon())
        {
            dir.normalize<3>();
            const Vector4 constraint = (curLength - m_length) * dir;
            const SimdFloat invCombinedMass(SimdFloat_1 / (m_massA + m_massB) * m_stiffness);
            *m_positionA += -m_massA * invCombinedMass * constraint;
            *m_positionB += m_massB * invCombinedMass * constraint;

            assert(!isnan((*m_positionA)(0)) && !isnan((*m_positionA)(1)) && !isnan((*m_positionA)(2)));
            assert(!isnan((*m_positionB)(0)) && !isnan((*m_positionB)(1)) && !isnan((*m_positionB)(2)));
        }
    }
}
