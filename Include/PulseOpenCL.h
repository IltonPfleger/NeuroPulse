#pragma once

#if defined(__PULSE_GPU_SUPPORTED)
#define __PULSE_GPU_CHECK(x) x
#else
#define __PULSE_GPU_CHECK(x) (printf("ERROR: PULSE GPUs Are Not Supported On This Device"), exit(1))
#endif

#ifdef __PULSE_GPU_SUPPORTED
#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

typedef struct
{
    cl_program PROGRAM;
    cl_kernel KERNEL;
} __PULSE_OPENCL_KERNEL;

static char RUNNING = 0;
static cl_device_id DEVICE_ID;
static cl_context CONTEXT;
static cl_command_queue QUEUE;
static size_t MAX_LOCAL_WORK_SIZE;

static cl_mem READ;
static cl_mem WRITE;

static __PULSE_OPENCL_KERNEL FeedDense;
static __PULSE_OPENCL_KERNEL BackDense;

static void __PULSE_OPENCL_CHECK_ERROR(cl_int err)
{
    if (err == CL_SUCCESS)
        return;
    printf("ERROR: OpenCL Code: %d.\n", err);
    exit(1);
}

static void __PULSE_OPENCL_GET_GPU()
{
    if (clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &DEVICE_ID, NULL) == CL_SUCCESS) ;
    {
        char device_name[128];
        clGetDeviceInfo(DEVICE_ID, CL_DEVICE_NAME, 128, &device_name, NULL);
        CONTEXT = clCreateContext(NULL, 1, &DEVICE_ID, NULL, NULL, NULL);
        QUEUE = clCreateCommandQueue(CONTEXT, DEVICE_ID, 0, NULL);
        printf("Using: %s.\n", device_name);
    }
};

static void __PULSE_OPENCL_GET_GPU_INFO()
{
    clGetDeviceInfo(DEVICE_ID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &MAX_LOCAL_WORK_SIZE, NULL);
};

static void __PULSE_OPENCL_COMPILE_KERNEL(__PULSE_OPENCL_KERNEL *kernel, const char *name, const char *src)
{
    kernel->PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char **)&src, NULL, NULL);
    cl_int err = clBuildProgram(kernel->PROGRAM, 0, NULL, NULL, NULL, NULL);
    kernel->KERNEL = clCreateKernel(kernel->PROGRAM, name, NULL);

    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(kernel->PROGRAM, DEVICE_ID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(kernel->PROGRAM, DEVICE_ID, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
    }
}

static void __PULSE_OPENCL_RELEASE_KERNEL(__PULSE_OPENCL_KERNEL kernel)
{
    __PULSE_OPENCL_CHECK_ERROR(clReleaseProgram(kernel.PROGRAM));
    __PULSE_OPENCL_CHECK_ERROR(clReleaseKernel(kernel.KERNEL));
}

cl_mem PULSE_OPENCL_ALLOC_READ_ONLY_MEM_ON_GPU(size_t size)
{
    return clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, size, NULL, NULL);
}

cl_mem PULSE_OPENCL_ALLOC_WRITE_ONLY_MEM_ON_GPU(size_t size)
{
    return clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY, size, NULL, NULL);
}

void PULSE_OPENCL_COPY_READ_MEM_TO_GPU(void *buffer, size_t offset, size_t size)
{
    clEnqueueWriteBuffer(QUEUE, READ, CL_TRUE, offset, size, buffer, 0, NULL, NULL);
}

void PULSE_OPENCL_COPY_WRITE_MEM_TO_GPU(void *buffer, size_t offset, size_t size)
{
    clEnqueueWriteBuffer(QUEUE, WRITE, CL_TRUE, offset, size, buffer, 0, NULL, NULL);
}

void PULSE_OPENCL_GET_WRITE_MEM_TO_HOST(void *buffer, size_t offset, size_t size)
{
    clEnqueueReadBuffer(QUEUE, WRITE, CL_TRUE, offset, size, buffer, 0, NULL, NULL);
}

void PULSE_OPENCL_ENQUEUE_FEEDDENSE(size_t i_size, size_t o_size)
{
    clSetKernelArg(FeedDense.KERNEL, 0, sizeof(cl_mem), &READ);
    clSetKernelArg(FeedDense.KERNEL, 1, sizeof(cl_mem), &WRITE);
    clSetKernelArg(FeedDense.KERNEL, 2, sizeof(int), (void *)&i_size);
    clSetKernelArg(FeedDense.KERNEL, 3, sizeof(int), (void *)&o_size);
    __PULSE_OPENCL_CHECK_ERROR(clEnqueueNDRangeKernel(QUEUE, FeedDense.KERNEL, 1, NULL, &o_size, NULL, 0, NULL, NULL));	//Local = NULL = OpenCL Will Find The Better Value
    clFinish(QUEUE);
}

void PULSE_OPENCL_ENQUEUE_BACKDENSE(size_t i_size, size_t o_size)
{
    clSetKernelArg(BackDense.KERNEL, 0, sizeof(cl_mem), &READ);
    clSetKernelArg(BackDense.KERNEL, 1, sizeof(cl_mem), &WRITE);
    clSetKernelArg(BackDense.KERNEL, 2, sizeof(int), (void *)&i_size);
    clSetKernelArg(BackDense.KERNEL, 3, sizeof(int), (void *)&o_size);
    __PULSE_OPENCL_CHECK_ERROR(clEnqueueNDRangeKernel(QUEUE, BackDense.KERNEL, 1, NULL, &o_size, NULL, 0, NULL, NULL));	//Local = NULL = OpenCL Will Find The Better Value
    clFinish(QUEUE);
}

static char * GetFeedDense_SRC()
{
    static const char FeedDense_SRC[] =
        "__kernel void FeedDense(__global float * A, __global float * B, const int i_size, const int o_size) {\n"
        "const int local_size = get_local_size(0);\n"
        "const int local_id = get_local_id(0);\n"
        "const int group_id = get_group_id(0);\n"
        "const int id = group_id*local_size + local_id; \n"
        "__global float * INPUTS = A;\n"
        "__global float * WEIGHTS = A + i_size;\n"
        "__global float * BAIASES = WEIGHTS + i_size*o_size;\n"
        "WEIGHTS += id*i_size; \n"
        "float acc = 0.0f; \n"
        "for(int i = 0; i < i_size; i++) { \n"
        "acc += INPUTS[i] * WEIGHTS[i]; \n"
        "};"
        "B[id] = acc + BAIASES[id]; \n"
        "};";

    /*"const int _local_size = get_local_size(0);\n"
      "const int _local_id = get_local_id(0);\n"
      "const int _group_id = get_group_id(0);\n"
      "const int id = _group_id*_local_size + _local_id;\n"
      "const int W_OFFSET = id*i_size; \n"
      "float acc = 0.0f; \n"
      "__local float local_a[_max_local_size]; \n"
      "for(int i = 0; i < i_size/_local_size; i++) { \n"
      "local_a[_local_id] = INPUTS[i*_local_size + _local_id]; \n"
      "barrier(CLK_LOCAL_MEM_FENCE); \n"
      "for(int j = 0; j < _local_size; j++) { \n"
      "acc += local_a[j] * WEIGHTS[W_OFFSET + i*_local_size + j]; \n" "}; \n"
      "barrier(CLK_LOCAL_MEM_FENCE); \n"
      "}; \n"
      "B[id] = acc + WEIGHTS[i_size*o_size + id]; \n"*/
    //"};";

    //static char KERNEL[sizeof(FeedDense_SRC) + 32];
    //sprintf(KERNEL, FeedDense_SRC, MAX_LOCAL_WORK_SIZE);
    return FeedDense_SRC;
}

static char *GetBackDense_SRC()
{
    static const char BackDense_SRC[] =
        "__kernel void BackDense(__global float * A, __global float * B, const int i_size, const int o_size) {\n" \
        "const int local_size = get_local_size(0);\n"
        "const int local_id = get_local_id(0);\n"
        "const int group_id = get_group_id(0);\n"
        "const int id = local_size*group_id + local_id;\n"
        "__global float * INPUTS = A;\n"
        "__global float * OUTPUTS = INPUTS + i_size;\n"
        "__global float * ERRORS = OUTPUTS + o_size;\n"
        "__global float * WEIGHTS = ERRORS + o_size;\n"
        "__global float * GRADIENTS = B;\n"
        "__global float * DELTAS = GRADIENTS + i_size*o_size;\n"
        "const float delta = ERRORS[id] * OUTPUTS[id]; \n"
        "DELTAS[id] += delta; \n"
        "for(int i = 0; i < i_size; i++) {\n"
        "GRADIENTS[id*i_size + i] += delta * INPUTS[i];"
        "}; \n"
        "};";

    return BackDense_SRC;

}

//static void _BackDense(PULSE_Layer *this)
//{
//    PULSE_DenseLayer dense = this->layer.DENSE;
//    this->activate(this->outputs, this->n_outputs, 1);
//    for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
//      PULSE_DataType delta = this->errors[i] * this->outputs[i];
//      dense.deltas[i] += delta;
//      for (int j = 0; j < this->n_inputs; j++) {
//          dense.gradients[wi + j] += delta * this->inputs[j];
//          if (this->parent != NULL)
//              this->parent->errors[j] += dense.weights[wi + j] * delta;
//      }
//    }
//}

void PULSE_OPENCL_START()
{
    if (!RUNNING)
    {
        __PULSE_OPENCL_GET_GPU();
        __PULSE_OPENCL_GET_GPU_INFO();
        __PULSE_OPENCL_COMPILE_KERNEL(&FeedDense, "FeedDense", GetFeedDense_SRC());
        __PULSE_OPENCL_COMPILE_KERNEL(&BackDense, "BackDense", GetBackDense_SRC());

        READ = PULSE_OPENCL_ALLOC_READ_ONLY_MEM_ON_GPU(5 * 1024 * 1024 * sizeof(float));
        WRITE = PULSE_OPENCL_ALLOC_WRITE_ONLY_MEM_ON_GPU(5 * 1024 * 1024 * sizeof(float));

        printf("WARNING: Pulse OpenCL Started!\n");
        RUNNING = 1;
    }
}

void PULSE_OPENCL_RELEASE()
{
    clFinish(QUEUE);
    __PULSE_OPENCL_RELEASE_KERNEL(FeedDense);
    clReleaseContext(CONTEXT);
    clReleaseCommandQueue(QUEUE);
    clReleaseDevice(DEVICE_ID);
    clReleaseMemObject(READ);
    clReleaseMemObject(WRITE);
    RUNNING = 0;
    printf("WARNING: Pulse OpenCL Finished!\n");
}

#endif
