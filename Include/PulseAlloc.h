#pragma once
#include "PulseTypes.h"
#include "PULSE_SIMD.h"

void * PULSE_Alloc(int , int , int , PULSE_OptimizationType);
void * PULSE_Alloc2D(int , int , int , int , PULSE_OptimizationType);
void PULSE_Free(void *);

