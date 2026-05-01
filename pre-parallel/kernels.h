/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

#include <cuda_runtime.h>

// Kernel function prototypes
//__global__ void test_kernel();

// Activation function and derivative
__device__
float relu_kernel(float x);
__device__
float drelu_kernel(float y);

__device__
float Tree_Sum(float *values);

__device__
void softmax_kernel(float *z, float *out, int len);

__global__
void train_model_parallel (MODEL* model, float* train_data, float* train_label, float* loss_array);
