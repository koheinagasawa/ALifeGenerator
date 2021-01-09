/*
* BaseType.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <cstdint>
#include <stdio.h>

#define WARN(str, ...) printf(str, __VA_ARGS__);

////////////

template <typename TYPE = uint32_t, int INVALID = -1>
struct TypedIndex
{
    inline void operator=(TYPE val) { m_val = val; }
    inline bool operator==(TYPE val) const { return m_val == val; }
    inline bool operator!=(TYPE val) const { return *this != val; }

    inline TYPE val() const { return m_val; }

    TYPE m_val;
};

#define DECLARE_ID_3_ARGS(Name, Type, Invalid) struct Name : TypedIndex<Type, Invalid> {};
#define DECLARE_ID_2_ARGS(Name, Type, ...) struct Name : TypedIndex<Type> {};
#define DECLARE_ID_1_ARG(Name, ...) struct Name : TypedIndex<> {};

// If the number of arguments passed to DECLARE_ID is N, then the 4th argument of this macro is going to be DECLARE_ID_N_ARGS
// assuming that the maximum number of arguments are 3.
#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define DECLARE_ID(...) GET_4TH_ARG(__VA_ARGS__, DECLARE_ID_3_ARGS, DECLARE_ID_2_ARGS, DECLARE_ID_1_ARG, )(__VA_ARGS__)

//////////////

