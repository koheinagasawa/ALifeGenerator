/*
* MassSpringSoftBodySystem.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Simulation/SoftBody/SoftBodySystem.h>
#include <Geometry/BasicTypes.h>
#include <vector>

class MassSpringSoftBodySystem : public SoftBodySystem
{
public:
    // Type Definitions
    using Points = std::vector<Point3D>;

protected:
    Points m_points;
};
