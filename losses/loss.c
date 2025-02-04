#include <losses/loss.h>

pulse_datatype pulse_mse(pulse_datatype *restrict x, pulse_datatype *restrict y, pulse_datatype *restrict errors, size_t size) {
    pulse_datatype loss = 0;
    for (int i = 0; i < size; i++) {
        errors[i] = 2 * (x[i] - y[i]);
        loss += pow(x[i] - y[i], 2);
    }
    return loss / size;
}

pulse_datatype pulse_msa(pulse_datatype *restrict x, pulse_datatype *restrict y, pulse_datatype *restrict errors, size_t size) {
    pulse_datatype loss = 0;
    for (int i = 0; i < size; i++) {
        errors[i] = fabs(x[i] - y[i]);
        loss += errors[i];
    }
    return loss / size;
}
