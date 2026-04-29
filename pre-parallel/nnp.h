/* nnp.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Header file for neural network model and training functions
*/

#ifndef NNP_H
#define NNP_H

// Model structure for neural network with two hidden layers
typedef struct tagMODEL{
    float W1[SIZE*H1];
    float b1[H1];
    float W2[H1*H2];
    float b2[H2];
    float W3[H2*CLASSES];
    float b3[CLASSES];
} MODEL;

//function prototypes
//nnp.cu specfic functions. There is a version in kernel and one here
void softmax_nnp(float *z, float *out, int len);
float relu_nnp(float x);
float drelu_nnp(float y);

void init_weights(float *w, int size);
void train_model(MODEL* model, float* train_data, float* train_label);
void save_model(MODEL* model);
void load_model(MODEL* model);
void predict(float *x, int i, MODEL* model);

#endif