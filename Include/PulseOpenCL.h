#pragma once
#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

typedef struct
{
	cl_program PROGRAM;
	cl_kernel KERNEL;
}__PULSE_OPENCL_KERNEL;


static cl_device_id DEVICE_ID;
static cl_context CONTEXT;
static cl_command_queue QUEUE;
static cl_mem MEM_A;
static cl_mem MEM_B;
static cl_mem MEM_C;
static size_t MAX_WORK_GROUP_SIZE;
static size_t BASE_BUFFER_SIZE = 512;
static cl_ulong GLOBAL_MEM_SIZE;
static __PULSE_OPENCL_KERNEL MM_1NxTMN;



static void __PULSE_OPENCL_CHECK_ERROR(cl_int err) {
	if (err == CL_SUCCESS) return;
	printf("ERROR: OpenCL Code: %d.\n", err);
	exit(1);
}

static void __PULSE_OPENCL_GET_GPU()
{
	if(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &DEVICE_ID, NULL) == CL_SUCCESS);
	{
		char device_name[128];
		clGetDeviceInfo(DEVICE_ID, CL_DEVICE_NAME, 128, &device_name, NULL);
		CONTEXT = clCreateContext(NULL, 1, &DEVICE_ID, NULL, NULL, NULL);
		QUEUE = clCreateCommandQueueWithProperties(CONTEXT, DEVICE_ID, NULL, NULL);
		printf("Using: %s.\n", device_name);
	}
};

static void __PULSE_OPENCL_GET_GPU_INFO()
{
	clGetDeviceInfo(DEVICE_ID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &MAX_WORK_GROUP_SIZE, NULL);
	clGetDeviceInfo(DEVICE_ID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &GLOBAL_MEM_SIZE, NULL);
};

static void __PULSE_OPENCL_FREE_GPU_BUFFERS()
{
	clReleaseMemObject(MEM_A);
	clReleaseMemObject(MEM_B);
	clReleaseMemObject(MEM_C);
}

static void __PULSE_OPENCL_ALLOC_GPU_BUFFERS()
{
	const int N = BASE_BUFFER_SIZE;
	__PULSE_OPENCL_FREE_GPU_BUFFERS();
	MEM_A = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, N*N*sizeof(float), NULL, NULL);
	MEM_B = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, N*N*sizeof(float), NULL, NULL);
	MEM_C = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY, N*N*sizeof(float),NULL, NULL);
}

static void __PULSE_OPENCL_COMPILE_KERNEL(__PULSE_OPENCL_KERNEL * kernel, const char * name, const char * src)
{
	kernel->PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char **)&src, NULL, NULL);
	__PULSE_OPENCL_CHECK_ERROR(clBuildProgram(kernel->PROGRAM, 0, NULL, NULL, NULL, NULL));
	kernel->KERNEL = clCreateKernel(kernel->PROGRAM, name, NULL);
}

static void __PULSE_OPENCL_FREE_KERNEL(__PULSE_OPENCL_KERNEL * kernel)
{
	__PULSE_OPENCL_CHECK_ERROR(clReleaseProgram(kernel->PROGRAM));
	__PULSE_OPENCL_CHECK_ERROR(clReleaseKernel(kernel->KERNEL));
}


static const char MM_1NxTMN_SRC[]  = "__kernel void MM_1NxTMN(const __global float * A, const __global float * B, __global float * C, const int N, const int M) {\n" \
																			"const int local_x = get_local_id(0);\n" \
																			"const int x = get_group_id(0);\n" \
																			"const int wi = (local_x + x) * N;\n" \
																			"const int LOCAL_SIZE = 512;\n"\
																			"__local float a[LOCAL_SIZE];\n"\
																			"float acc = 0.0f;\n" \
																			"if(x*LOCAL_SIZE + local_x < N) {\n" \
																			"for(int i = 0; i < N/LOCAL_SIZE; i++) {\n" \
																			"a[local_x] = A[i*LOCAL_SIZE + local_x];\n"\
																			"barrier(CLK_LOCAL_MEM_FENCE);\n"\
																			"for(int j = 0; j < LOCAL_SIZE; j++) {\n" \
																			"acc += a[j] * B[wi + i*LOCAL_SIZE + j];\n"\
																			"};"\
																			"barrier(CLK_LOCAL_MEM_FENCE);\n"\
																			"};"\
																			"C[x*LOCAL_SIZE + local_x] += acc;\n"\
																			"};"\
																			"};";





void __PULSE_OPENCL_START()
{
	static char STARTED = 0;
	if(!STARTED)
	{
		__PULSE_OPENCL_GET_GPU();
		__PULSE_OPENCL_GET_GPU_INFO();
		__PULSE_OPENCL_ALLOC_GPU_BUFFERS();
		__PULSE_OPENCL_COMPILE_KERNEL(&MM_1NxTMN, "MM_1NxTMN", MM_1NxTMN_SRC);
		STARTED = 1;
	}
}


void __PULSE_OPENCL_MM_1NxTMN(float * A, float * B, float * BETA, float * dst, int N, int M)
{
	size_t global[] = {M};
	size_t local[] = {512};
	__PULSE_OPENCL_CHECK_ERROR(clEnqueueWriteBuffer(QUEUE, MEM_A, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL));
	__PULSE_OPENCL_CHECK_ERROR(clEnqueueWriteBuffer(QUEUE, MEM_B, CL_TRUE, 0, N * M * sizeof(float), B, 0, NULL, NULL));
	if(BETA != NULL)
		__PULSE_OPENCL_CHECK_ERROR(clEnqueueWriteBuffer(QUEUE, MEM_C, CL_TRUE, 0, M * sizeof(float), BETA, 0, NULL, NULL));

	clSetKernelArg(MM_1NxTMN.KERNEL, 0, sizeof(cl_mem), (void*)&MEM_A);
	clSetKernelArg(MM_1NxTMN.KERNEL, 1, sizeof(cl_mem), (void*)&MEM_B);
	clSetKernelArg(MM_1NxTMN.KERNEL, 2, sizeof(cl_mem), (void*)&MEM_C);
	clSetKernelArg(MM_1NxTMN.KERNEL, 3, sizeof(int), (void*)&N);
	clSetKernelArg(MM_1NxTMN.KERNEL, 4, sizeof(int), (void*)&M);
	cl_event event;
	__PULSE_OPENCL_CHECK_ERROR(clEnqueueNDRangeKernel(QUEUE, MM_1NxTMN.KERNEL, 1, NULL, global, local, 0, NULL, &event));
	__PULSE_OPENCL_CHECK_ERROR(clWaitForEvents(1, &event));
	__PULSE_OPENCL_CHECK_ERROR(clEnqueueReadBuffer(QUEUE, MEM_C, CL_TRUE, 0, M * sizeof(float), dst, 0, NULL, NULL));
}
