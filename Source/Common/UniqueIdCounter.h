/*
* Genome.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <iostream>

// Helper class to increment unique id.
template <typename T>
class UniqueIdCounter
{
public:
    UniqueIdCounter() = default;

    // Returns a new innovation id. New id will be returned every time you call this function.
    T getNewId()
    {
        T idOut = m_nextId;
        m_nextId = m_nextId.val() + 1;

        if (m_nextId == 0)
        {
            std::cerr << "ERROR: Unique ID overflowed!" << std::endl;
        }

        return idOut;
    }

    void reset() { m_nextId = 0; }

protected:
    UniqueIdCounter(const UniqueIdCounter&) = delete;
    void operator=(const UniqueIdCounter&) = delete;

    T m_nextId = 0;
};
