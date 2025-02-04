// #include <llayer.h>
//
// pulse_layer_t pulse_create_recurrent_layer(size_t n_inputs, size_t n_outputs, pulse_activation_fnc_e activation, pulse_optimization_e optimization)
// {
//     pulse_layer_t layer;
//
//     layer.n_inputs  = n_inputs;
//     layer.n_outputs = n_outputs;
//
//     layer.inputs  = pulse_memory_alloc(sizeof(pulse_data_type) * n_inputs);
//     layer.outputs = pulse_memory_alloc(sizeof(pulse_data_type) * n_outputs);
//     layer.w       = pulse_memory_alloc(sizeof(pulse_data_type) * n_inputs * n_outputs);
//
//     layer.type         = PULSE_RECURRENT;
//     layer.optimization = optimization;
//     layer.activate     = pulse_get_activation_fnc_ptr(activation);
//
//     layer.prev = NULL;
//     layer.next = NULL;
//
//     // layer.n_weights    = (n_outputs * n_inputs) + n_outputs;
//     layer.free = NULL;
//
//     for (int i = 0; i < n_inputs * n_outputs; i++) layer.w[i] = ((pulse_data_type)rand() / (pulse_data_type)(RAND_MAX)) * 0.01;
//
//     return layer;
// }
