/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"

// tree reduction for array values
__device__
float Tree_Sum(float *values) {
    int i = threadIdx.x % blockDim.x;
    // <<=1 is left shift by 1, which is basically multiplication by 2
    for (int stride = 1; stride < blockDim.x; stride<<=1 ) {
        if (i % (stride << 1) == 0) values[i] += values[i+stride];
        __syncthreads();
    }

    return values[i];
}

/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
__device__
float relu_kernel(float x) { return x > 0 ? x : 0; }

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
__device__
float drelu_kernel(float y) { return y > 0 ? 1 : 0; }

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
__device__
void softmax_kernel(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

__global__
void train_model_parallel (MODEL* model, float* train_data, float* train_label, float* loss_array) {
    // calculate which thread this is, only a part of these threads will run the function
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread index: %d\n", n);

    if (n < NUM_TRAIN) {

        // array to store loss values from same thread block
        __shared__ float shared_loss[1024];

        // ---------- Forward ----------
        float h1[H1], h1a[H1];
        for (int j=0;j<H1;j++){
            h1[j]=model->b1[j];
            for (int i=0;i<SIZE;i++) h1[j]+=train_data[n * SIZE + i]*model->W1[i*H1+j];
            h1a[j]=relu_kernel(h1[j]);
        }
        float h2[H2], h2a[H2];
        for (int j=0;j<H2;j++){
            h2[j]=model->b2[j];
            for (int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j];
            h2a[j]=relu_kernel(h2[j]);
        }
        float out[CLASSES], outa[CLASSES];
        for (int k=0;k<CLASSES;k++){
            out[k]=model->b3[k];
            for (int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k];
        }
        softmax_kernel(out,outa,CLASSES);

        // ---------- Loss ----------
        // initalize shared_loss array to first class
        shared_loss[n] = train_label[n * CLASSES + 0]*logf(outa[0]+1e-8f);
        // now sum the losses
        for (int k=1;k<CLASSES;k++)
            shared_loss[n] -= train_label[n * CLASSES + k]*logf(outa[k]+1e-8f);
        __syncthreads();

        // store sum of array in here, but only return the one in thread 0
        float loss_sum = Tree_Sum(shared_loss);
        if (threadIdx.x % blockDim.x == 0) {
            atomicAdd(&loss_array[blockIdx.x], loss_sum);
        }

        // ---------- Backprop ----------
        float delta3[CLASSES];
        for (int k=0;k<CLASSES;k++)
            delta3[k] = train_label[n * CLASSES + k]-outa[k];

        float delta2[H2];
        for (int j=0;j<H2;j++){
            float err=0;
            for (int k=0;k<CLASSES;k++) err+=delta3[k]*model->W3[j*CLASSES+k];
            delta2[j]=err*drelu_kernel(h2a[j]);
        }

        float delta1[H1];
        for (int j=0;j<H1;j++){
            float err=0;
            for (int k=0;k<H2;k++) err+=delta2[k]*model->W2[j*H2+k];
            delta1[j]=err*drelu_kernel(h1a[j]);
        }

        // ---------- Update ----------
        for (int j=0;j<H2;j++)
            for (int k=0;k<CLASSES;k++)
                model->W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
        for (int k=0;k<CLASSES;k++) model->b3[k]+=LR*delta3[k];

        for (int j=0;j<H1;j++)
            for (int k=0;k<H2;k++)
                model->W2[j*H2+k]+=LR*delta2[k]*h1a[j];
        for (int k=0;k<H2;k++) model->b2[k]+=LR*delta2[k];

        for (int i=0;i<SIZE;i++)
            for (int j=0;j<H1;j++)
                model->W1[i*H1+j]+=LR*delta1[j]*train_data[n * SIZE + i];
        for (int j=0;j<H1;j++) model->b1[j]+=LR*delta1[j];
    }
}