/*
 * loader.c
 *
 *  Created on: Nov 9, 2025
 *
 *  Functions to load the MNIST dataset from binary files into memory.
 *  The dataset consists of training and testing images and their corresponding labels.
 *  Images are normalized to the range [0,1] and labels are one-hot encoded.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"

// Helper: read big-endian 32-bit int
int read_int(FILE *f)
{
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

/* load_data: load image data from binary file
 * Arguments:
 *   filename: path to the binary data file
 *   data: 2D array to store loaded image data
 *   num: number of images to load
 */
void load_data_train(const char *filename, float *train_data, int num)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        perror("open data file");
        exit(1);
    }
    read_int(f); // magic number
    int n = read_int(f);
    int rows = read_int(f);
    int cols = read_int(f);
    if (n < num && num != 0)
        num = n;
    unsigned char *buf = (unsigned char *)malloc(rows * cols);
    for (int i = 0; i < num; i++)
    {
        fread(buf, 1, rows * cols, f);
        for (int j = 0; j < rows * cols; j++)
        {
            // printf("%d\n", buf[0]);
            train_data[i * SIZE + j] = buf[j] / 255.0f; // normalize to [0,1]
            // printf("good here!\n");
        }
    }
    free(buf);
    fclose(f);
}
void load_data_predict(const char *filename, float *predict_data, int num)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        perror("open data file");
        exit(1);
    }
    read_int(f); // magic number
    int n = read_int(f);
    int rows = read_int(f);
    int cols = read_int(f);
    if (n < num && num != 0)
        num = n;
    unsigned char *buf = (unsigned char *)malloc(rows * cols);
    for (int i = 0; i < num; i++)
    {
        fread(buf, 1, rows * cols, f);
        for (int j = 0; j < rows * cols; j++)
        {
            // printf("good here!\n");
            predict_data[i * SIZE + j] = buf[j] / 255.0f; // normalize to [0,1]
        }
    }
    free(buf);
    fclose(f);
}

/* load_labels: load label data from binary file
 * Arguments:
 *   filename: path to the binary label file
 *   labels: 2D array to store loaded one-hot encoded labels
 *   num: number of labels to load
 */
void load_labels_train(const char *filename, float *train_labels, int num)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        perror("open label file");
        exit(1);
    }
    read_int(f); // magic number
    int n = read_int(f);
    if (n < num && num != 0)
        num = n;
    unsigned char *buf = (unsigned char *)malloc(n);
    fread(buf, 1, n, f);
    for (int i = 0; i < num; i++)
    {
        for (int k = 0; k < CLASSES; k++)
            train_labels[i * CLASSES + k] = 0.0f;
        train_labels[i * CLASSES + buf[i]] = 1.0f; // one-hot encode
    }
    free(buf);
    fclose(f);
}
void load_labels_predict(const char *filename, float *predict_labels, int num)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        perror("open label file");
        exit(1);
    }
    read_int(f); // magic number
    int n = read_int(f);
    if (n < num && num != 0)
        num = n;
    unsigned char *buf = (unsigned char *)malloc(n);
    fread(buf, 1, n, f);
    for (int i = 0; i < num; i++)
    {
        for (int k = 0; k < CLASSES; k++)
            predict_labels[i * CLASSES + k] = 0.0f;
        predict_labels[i * CLASSES + buf[i]] = 1.0f; // one-hot encode
    }
    free(buf);
    fclose(f);
}

/* load_dataset: load the entire MNIST dataset into memory (training version and predict version)
 * Arguments:
 *   None
 */
void load_dataset_train(float *train_data, float *train_label)
{
    load_data_train(TRAIN_DATA, train_data, NUM_TRAIN);
    load_labels_train(TRAIN_LABELS, train_label, NUM_TRAIN);
}

void load_dataset_predict(float *test_data, float *test_label)
{
    load_data_predict(TEST_DATA, test_data, NUM_TEST);
    load_labels_predict(TEST_LABELS, test_label, NUM_TEST);
}
