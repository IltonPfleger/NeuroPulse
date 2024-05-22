#ifndef LAYERS_H
#define LAYERS_H
#include "Activations.h" 

struct PULSE_LayerStruct;
typedef void (*PULSE_FowardLayerFunctionPtr)(struct PULSE_LayerStruct *, double *);
typedef void (*PULSE_BackLayerFunctionPtr)(struct PULSE_LayerStruct *);
typedef void (*PULSE_FixLayerFunctionPtr)(struct PULSE_LayerStruct *, int, double);
typedef void (*PULSE_ActivationFunctionPtr)(double *, int, int);

typedef struct PULSE_LayerStruct
{
	int n;
	int size;
	double *weights;
	double *gradients;
	double *baias;
	double *inputs;
	double *outputs;
	double *deltas;
	double *error;
	struct PULSE_LayerStruct *parent;
	struct PULSE_LayerStruct *child;
	PULSE_ActivationFunctionPtr activate;
	PULSE_FowardLayerFunctionPtr feed;
	PULSE_BackLayerFunctionPtr back;
	PULSE_FixLayerFunctionPtr fix;
} PULSE_Layer;


PULSE_Layer PULSE_CreateLayer(int, int, PULSE_ActivationFunctionPtr);
static void PULSE_FowardLayer(PULSE_Layer *, double *);
static void PULSE_BackLayer(PULSE_Layer *);
static void PULSE_FixLayer(PULSE_Layer*, int, double);

#endif
