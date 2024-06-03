#ifndef _PULSE_LAYER
#define _PULSE_LAYER

struct PULSE_LayerStruct;
typedef struct {int batch_size; double lr;} PULSE_HyperArgs; 
typedef double PULSE_DataType;

typedef void (*PULSE_FeedLayerFunctionPtr)(struct PULSE_LayerStruct *);
typedef void (*PULSE_BackLayerFunctionPtr)(struct PULSE_LayerStruct *);
typedef void (*PULSE_ActivationLayerFunctionPtr)(struct PULSE_LayerStruct *, char);
typedef void (*PULSE_FixLayerFunctionPtr)(struct PULSE_LayerStruct *, PULSE_HyperArgs);
typedef void (*PULSE_DestroyLayerFunctionPtr)(struct PULSE_LayerStruct *);

typedef enum
{
	PULSE_NONE,
	PULSE_DENSE,
	PULSE_CONV,
	PULSE_MAXPOLL
} PULSE_LayerType;

typedef struct PULSE_LayerStruct
{
	PULSE_LayerType type;
	PULSE_DataType *inputs;
	PULSE_DataType *outputs;
	PULSE_DataType *errors;
	PULSE_FeedLayerFunctionPtr feed;
	PULSE_BackLayerFunctionPtr back;
	PULSE_FixLayerFunctionPtr fix;
	PULSE_DestroyLayerFunctionPtr destroy;
	PULSE_ActivationLayerFunctionPtr activate;
	struct PULSE_LayerStruct * parent;
	struct PULSE_LayerStruct * child;
	void * layer;
	unsigned int n_inputs;
	unsigned int n_outputs;
} PULSE_Layer;


PULSE_Layer PULSE_CreateLayer(int n_inputs, int n_outputs, PULSE_LayerType type, PULSE_FeedLayerFunctionPtr feed, PULSE_BackLayerFunctionPtr back, PULSE_FixLayerFunctionPtr fix, PULSE_DestroyLayerFunctionPtr destroy);
void PULSE_DestroyLayer();


#endif
