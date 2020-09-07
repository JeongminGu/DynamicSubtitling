#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"
#include <process.h>

struct svm_node *x;
struct svm_model *model;
int predict_probability = 0;
int x_end = 24;
int set_trig = 0;
int svm_type = 0;
int nr_class = 0;
double predict_label;

//threading model loading
void model_load()
{
	const char *model_path = "Train_all.model";
	model = svm_load_model(model_path);
	svm_type = svm_get_svm_type(model); // SVM Type : enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR }
	nr_class = svm_get_nr_class(model); // model has 3 classes
}

double SVM_Classifier(float* distances)
{
	if (set_trig == 0)
	{
		model_load();
	}

	double *prob_estimates = NULL;
	double target_label = 1.0; // In order to operate well, I set the constant target number.
	double predict_label;
	//char *label = {'1', };

	x = (struct svm_node *) malloc(25 * sizeof(struct svm_node));

	for (unsigned int long i = 0; i < 24; i++)
	{
		x[i].index = i+1;
		x[i].value = distances[i];
	}

	x[x_end].index = -1;

	if (predict_probability && (svm_type == C_SVC || svm_type == NU_SVC))
	{
		predict_label = svm_predict_probability(model, x, prob_estimates);
	}
	else
	{
		predict_label = svm_predict(model, x);

	}

	if (predict_probability)
	{
		free(prob_estimates);
	}
	svm_free_and_destroy_model(&model);
	free(x);

	return predict_label;
}

void kkkk38(double* result, double** arr, int faceNum, int face_index, struct svm_model* model)
{
	double* prob_estimates = NULL;
	double target_label = 1.0;

	struct svm_node** xarr = (struct  svm_node**)malloc(sizeof((*arr)[0]) * sizeof(struct svm_node**));
	for (int i = 0; i < sizeof((*arr)[0]); i++)
		xarr[i] = (struct svm_node*) malloc(25 * sizeof(struct svm_node));
	//x = (struct svm_node *) malloc(25 * sizeof(struct svm_node));

	for (int j = 0; j < faceNum; ++j) {
		for (unsigned int long i = 0; i < 20; i++)
		{
			xarr[j][i].index = i + 1;
			xarr[j][i].value = (double)arr[j][i];
		}
		xarr[j][x_end].index = -1;
	}

	predict_label = svm_predict(model, xarr[face_index]);
	result[face_index] = predict_label;

	free(x);
	free(xarr);
}

