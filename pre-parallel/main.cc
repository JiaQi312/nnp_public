/*
 * main.cc
 *
 *  Created on: Nov 9, 2025
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include <cuda_runtime.h>

/* Command definitions for command line options */
#define TRAIN 1
#define PREDICT 2

/* parseCmd: parse command line arguments
 * Arguments:
 *   argc: argument count
 *   argv: argument vector
 * Returns:
 *   TRAIN or PREDICT based on user input, 0 for invalid input
 */
int parseCmd(int argc, char **argv)
{
	if (argc != 2)
		return 0;
	if (strcmp(argv[1], "train") == 0)
		return TRAIN;
	if (strcmp(argv[1], "predict") == 0)
		return PREDICT;
	return 0;
}

/* usage: print usage information
 * Returns:
 *   0
 */
int usage()
{
	printf("Usage: nnp [train|predict]\n\tNote: predict requires a previously trained model in the directory named model.bin\n");
	return 0;
}

/* train: load dataset, train model, save model
 * Returns:
 *   void
 */
void train(MODEL *model, float *train_data, float *train_label)
{

	load_dataset_train(train_data, train_label);
	// printf("good here!\n");
	time_t t = time(NULL);
	train_model(model, train_data, train_label);
	t = time(NULL) - t;
	save_model(model);
	printf("Trained model in %ld seconds\n", t);
}
/* predict_test: load dataset, load model, predict on test set
 * Returns:
 *   void
 */
void predict_test(float *test_data, float *test_label)
{
	load_dataset_predict(test_data, test_label);
	MODEL model;
	load_model(&model);
	for (int i = 0; i < NUM_TEST; i++)
	{
		predict(test_data, i, &model);
	}
}

/* main: entry point of the program
 * Arguments:
 *   argc: argument count
 *   argv: argument vector
 * Returns:
 *   0 on success, usage information on invalid input
 */
int main(int argc, char **argv)
{

	// make model, training data, and testing data accessible by both cpu and gpu
	MODEL *model;
	float *train_data;
	float *train_label;
	float *test_data;
	float *test_label;
	cudaMallocManaged(&model, sizeof(MODEL));
	cudaError_t err = cudaMallocManaged(&model, sizeof(MODEL));
	if (err != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return -1;
	}

	cudaMallocManaged(&train_data, sizeof(float) * NUM_TRAIN * SIZE);
	cudaMallocManaged(&train_label, sizeof(float) * NUM_TRAIN * CLASSES);
	cudaMallocManaged(&test_data, sizeof(float) * NUM_TEST * SIZE);
	cudaMallocManaged(&test_label, sizeof(float) * NUM_TEST * CLASSES);

	// initialize arrays to 0
	for (int i = 0; i < NUM_TRAIN * SIZE; i++)
	{
		train_data[i] = 0.0f;
	}
	for (int i = 0; i < NUM_TRAIN * CLASSES; i++)
	{
		train_label[i] = 0.0f;
	}
	for (int i = 0; i < NUM_TEST * SIZE; i++)
	{
		test_data[i] = 0.0f;
	}
	for (int i = 0; i < NUM_TEST * CLASSES; i++)
	{
		test_label[i] = 0.0f;
	}

	switch (parseCmd(argc, argv))
	{
	case TRAIN:
	{
		train(model, train_data, train_label);
		break;
	}
	case PREDICT:
	{
		predict_test(train_data, train_label);
		break;
	}
	default:
	{
		return usage();
	}
	}

	// free cudaManaged model
	cudaFree(model);
	cudaFree(train_data);
	cudaFree(train_label);
	cudaFree(test_data);
	cudaFree(test_label);
}
