/* loader.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Header file for data loading functions
*/
#ifndef LOADER_H
#define LOADER_H

// Helper: read big-endian 32-bit int
void load_data_train(const char *filename, float* train_data, int num); 
void load_labels_train(const char *filename, float* train_labels, int num);
void load_dataset_train(float* train_data, float* train_label);

void load_data_predict(const char *filename, float* predict_data, int num); 
void load_labels_predict(const char *filename, float* predict_labels, int num);
void load_dataset_predict(float* train_data, float* train_label);
#endif
