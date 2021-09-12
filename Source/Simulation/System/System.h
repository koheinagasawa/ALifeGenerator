/*
* System.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

class System
{
public:
    virtual void step(float deltaTime) = 0;
};
