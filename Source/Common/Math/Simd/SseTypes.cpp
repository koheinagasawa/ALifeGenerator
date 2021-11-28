/*
* SseTypes.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Common/Common.h>
#include <Common/Math/Simd/SseTypes.h>

#ifdef USE_SSE

extern const QuadrupleFloat QuadFloat1111 = _mm_set_ps(1, 1, 1, 1);

#endif
