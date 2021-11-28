/*
* Collider.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Geometry/Shapes/Shape.h>

#include <memory>

// A type representing a shaped object which collides against something during simulation.
class Collider
{
public:
    // Type Definition
    using ShapePtr = std::shared_ptr<Shape>;

    Collider(ShapePtr shape) : m_shape(shape) {}

    const Shape* getShape() const { return m_shape.get(); }

protected:
    ShapePtr m_shape;
};
