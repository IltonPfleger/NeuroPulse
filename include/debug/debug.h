#ifndef PULSE_DEBUG_H
#define PULSE_DEBUG_H
#include <stdio.h>

#define PULSE_DEBUG_ERROR_ENABLED
// #define PULSE_DEBUG_LOGGER_ENABLED

#ifdef PULSE_DEBUG_ERROR_ENABLED
#define PULSE_DEBUG_ERROR(EXPRESSION, LOG)                                                \
    do {                                                                                  \
        if (EXPRESSION) {                                                                 \
            fprintf(stderr, "PULSE[ERROR] >> { %s } [%s,%d]\n", LOG, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)
#else
#define PULSE_DEBUG_ERROR(EXPRESSION, LOG)
#endif

#ifdef PULSE_DEBUG_LOGGER_ENABLED
#define PULSE_DEBUG_LOGGER(LOG, ...) fprintf(stdout, "PULSE[LOGGER] >> " LOG, ##__VA_ARGS__);
#else
#define PULSE_DEBUG_LOGGER(LOG, ...)
#endif

#endif
