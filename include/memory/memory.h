#ifndef __MEMORY__
#define __MEMORY__

#include <stddef.h>
#include <stdlib.h>

inline void* pulse_memory_alloc(size_t size) { return malloc(size); };
inline void pulse_memory_free(void* ptr) { free(ptr); };

#endif
