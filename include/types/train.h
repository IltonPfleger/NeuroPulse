#ifndef PULSE_TYPES_TRAIN_H
#define PULSE_TYPES_TRAIN_H

typedef struct {
    size_t samples;
    size_t batch_size;
    size_t epoch;
    double lr;
} pulse_train_args_t;

#endif
