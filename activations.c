#include <activations.h>

static void PULSE_Sigmoid(PULSE_DATA* x, size_t size, char prime) {
    for (int i = 0; i < size; i++) {
        if (prime) x[i] = x[i] * (1.f - x[i]);
        x[i] = 1.f / (1.f + expf(-x[i]));
    }
}

static void PULSE_ReLU(PULSE_DATA* x, size_t size, char prime) {
    for (int i = 0; i < size; i++) {
        if (prime) x[i] = x[i] > 0 ? 1 : 0;
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

static void PULSE_LeakyReLU(PULSE_DATA* x, size_t size, char prime) {
    for (int i = 0; i < size; i++) {
        if (prime) x[i] = x[i] > 0 ? 1 : 0.0001;
        x[i] = x[i] > 0 ? x[i] : 0.0001 * x[i];
    }
}

void* pulse_get_activation_fnc_ptr(pulse_activation_fnc_e type) {
    switch (type) {
        case PULSE_ACTIVATION_NONE:
            return NULL;
        case PULSE_ACTIVATION_SIGMOID:
            return PULSE_Sigmoid;
        case PULSE_ACTIVATION_RELU:
            return PULSE_ReLU;
        case PULSE_ACTIVATION_LEAKYRELU:
            return PULSE_LeakyReLU;
    }
    return NULL;
}
