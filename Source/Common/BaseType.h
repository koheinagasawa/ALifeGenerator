/*
* BaseType.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <cstdint>
#include <stdio.h>

#define WARN(str, ...) printf(str##"\n", __VA_ARGS__);

////////////

template <typename TYPE = uint32_t, int INVALID = -1>
struct TypedIndex
{
    // Constructors
    TypedIndex() : m_val(INVALID) {}
    TypedIndex(TYPE val) : m_val(val) {}
    TypedIndex(const TypedIndex& other) : m_val(other.m_val) {}

    // Operators
    inline void operator=(TYPE val) { m_val = val; }
    inline void operator=(const TypedIndex& other) { m_val = other.m_val; }
    inline bool operator==(TYPE val) const { return m_val == val; }
    inline bool operator==(const TypedIndex& other) const { return m_val == other.m_val; }
    inline bool operator!=(TYPE val) const { return !(*this == val); }
    inline bool operator!=(const TypedIndex& other) const { return !(*this == other); }
    inline bool operator<(const TypedIndex& other) const { return m_val < other.m_val; }
    inline bool operator<=(const TypedIndex& other) const { return m_val <= other.m_val; }
    inline bool operator>(const TypedIndex& other) const { return m_val > other.m_val; }
    inline bool operator>=(const TypedIndex& other) const { return m_val >= other.m_val; }

    inline bool isValid() const { return m_val != INVALID; }

    inline TYPE val() const { return m_val; }

    TYPE m_val;
};

#define DECLARE_TYPED_INDEX_HASHER(Name, Type) namespace std { template<> struct hash<Name> { std::size_t operator()(const Name& id) const { return hash<Type>()(id.m_val); } }; }
#define DECLARE_TYPED_INDEX_HASHER_2_ARGS(Name, Type) DECLARE_TYPED_INDEX_HASHER(Name, Type)
#define DECLARE_TYPED_INDEX_HASHER_1_ARG(Name) DECLARE_TYPED_INDEX_HASHER(Name, uint32_t)

#define DECLARE_TYPED_INDEX_CONSTRUCTORS(Name, Type, Invalid) \
    Name() : TypedIndex() {} \
    Name(Type val) : TypedIndex(val) {} \
    Name(const Name& other) : TypedIndex(other) {} \
    static inline Name invalid() { return Name(Invalid); }

#define DECLARE_ID_3_ARGS(Name, Type, Invalid) \
    struct Name : TypedIndex<Type, Invalid> { DECLARE_TYPED_INDEX_CONSTRUCTORS(Name, Type, Invalid); }; \
    DECLARE_TYPED_INDEX_HASHER_2_ARGS(Name, Type)
#define DECLARE_ID_2_ARGS(Name, Type) \
    struct Name : TypedIndex<Type> { DECLARE_TYPED_INDEX_CONSTRUCTORS(Name, Type, -1); }; \
    DECLARE_TYPED_INDEX_HASHER_2_ARGS(Name, Type)
#define DECLARE_ID_1_ARG(Name) \
    struct Name : TypedIndex<> { DECLARE_TYPED_INDEX_CONSTRUCTORS(Name, uint32_t, -1); }; \
    DECLARE_TYPED_INDEX_HASHER_1_ARG(Name)

// If the number of arguments passed to DECLARE_ID is N, then the 4th argument of this macro is going to be DECLARE_ID_N_ARGS
// assuming that the maximum number of arguments are 3.
#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
// Macro to just pass whatever it received. This is needed for the macro below to prevent the inner macro name doesn't not get expanded in msvc.
#define PASS_ON(...) __VA_ARGS__
#define DECLARE_ID(...) PASS_ON(PASS_ON(GET_4TH_ARG(__VA_ARGS__, DECLARE_ID_3_ARGS, DECLARE_ID_2_ARGS, DECLARE_ID_1_ARG, ))(__VA_ARGS__))

//////////////

