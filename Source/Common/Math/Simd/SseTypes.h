/*
* SseTypes.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/BaseType.h>

#ifdef USE_SSE

#include <xmmintrin.h>
#include <immintrin.h>

using SingleFloat = __m128;
using QuadrupleFloat = __m128;

#else

using SingleFloat = float;
struct QuadrupleFloat { float m_floats[4]; };

#endif
