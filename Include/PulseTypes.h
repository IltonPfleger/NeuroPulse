#ifndef __PULSE_TYPES__
#define __PULSE_TYPES__
#define PULSE_DataType float
#define PULSE_N unsigned int
#define PULSE_Bool char
#define PULSE_Void void

typedef struct{
	int batch_size;
	float lr;
} PULSE_HyperArgs; 

struct PULSE_Layer;
typedef void (*PULSE_FeedLayerFunctionPtr)(struct PULSE_Layer *);
typedef void (*PULSE_BackLayerFunctionPtr)(struct PULSE_Layer *);
typedef void (*PULSE_FixLayerFunctionPtr)(struct PULSE_Layer *, PULSE_HyperArgs);
typedef void (*PULSE_DestroyLayerFunctionPtr)(struct PULSE_Layer *);
typedef void (*PULSE_ActivationLayerFunctionPtr)(PULSE_DataType *, PULSE_N, char);

typedef struct {
	PULSE_DataType * weights;
	PULSE_DataType * baias;
	PULSE_DataType * deltas;
	PULSE_DataType * gradients;
}PULSE_DenseLayer;

struct PULSE_Layer;
typedef void (*PULSE_FeedLayerFunctionPtr)(struct PULSE_Layer *);
typedef void (*PULSE_BackLayerFunctionPtr)(struct PULSE_Layer *);
typedef void (*PULSE_FixLayerFunctionPtr)(struct PULSE_Layer *, PULSE_HyperArgs);
typedef void (*PULSE_DestroyLayerFunctionPtr)(struct PULSE_Layer *);
typedef void (*PULSE_ActivationLayerFunctionPtr)(struct PULSE_Layer *, char);

typedef enum
{
	PULSE_NONE,
	PULSE_DENSE,
	PULSE_CONV,
	PULSE_MAXPOLL
} PULSE_LayerType;

typedef struct PULSE_Layer
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
	PULSE_Void * layer;
	PULSE_N n_inputs;
	PULSE_N n_outputs;
	struct PULSE_Layer * parent;
	struct PULSE_Layer * child;
} PULSE_Layer;


#endif
