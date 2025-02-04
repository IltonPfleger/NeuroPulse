#include <activations/activations.h>

void pulse_sigmoid(pulse_datatype* x, size_t size, char prime) {
    for (int i = 0; i < size; i++) {
        if (prime) x[i] = x[i] * (1.f - x[i]);
        x[i] = 1.f / (1.f + expf(-x[i]));
    }
}

void pulse_relu(pulse_datatype* x, size_t size, char prime) {
    for (int i = 0; i < size; i++) {
        if (prime) x[i] = x[i] > 0 ? 1 : 0;
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void pulse_leaky_relu(pulse_datatype* x, size_t size, char prime) {
    for (int i = 0; i < size; i++) {
        if (prime) x[i] = x[i] > 0 ? 1 : 0.0001;
        x[i] = x[i] > 0 ? x[i] : 0.0001 * x[i];
    }
}
