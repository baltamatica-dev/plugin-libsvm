#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm.h"

#include "libsvm.hpp"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048

int print_null(const char *s,...) {return 0;}
int (*info)(const char *fmt,...) = &bxPrintf;

void read_sparse_instance(const bxArray *prhs, int index, struct svm_node *x)
{
	int i, j, low, high;
	baIndex *ir, *jc;
	double *samples;

	ir = bxGetIr(prhs);
	jc = bxGetJc(prhs);
	samples = bxGetPr(prhs);

	// each column is one instance
	j = 0;
	low = (int)jc[index], high = (int)jc[index+1];
	for(i=low;i<high;i++)
	{
		x[j].index = (int)ir[i] + 1;
		x[j].value = samples[i];
		j++;
	}
	x[j].index = -1;
}

static void fake_answer(int nlhs, bxArray *plhs[])
{
	int i;
	for(i=0;i<nlhs;i++)
		plhs[i] = bxCreateDoubleMatrix(0, 0, bxREAL);
}

void predict(int nlhs, bxArray *plhs[], const bxArray *prhs[], struct svm_model *model, const int predict_probability)
{
	int label_vector_row_num, label_vector_col_num;
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_instance, *ptr_label, *ptr_predict_label;
	double *ptr_prob_estimates, *ptr_dec_values, *ptr;
	struct svm_node *x;
	bxArray *pplhs[1]; // transposed instance sparse matrix
	bxArray *tplhs[3]; // temporary storage for plhs[]

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;

	// prhs[1] = testing instance matrix
	feature_number = (int)bxGetN(prhs[1]);
	testing_instance_number = (int)bxGetM(prhs[1]);
	label_vector_row_num = (int)bxGetM(prhs[0]);
	label_vector_col_num = (int)bxGetN(prhs[0]);

	if(label_vector_row_num!=testing_instance_number)
	{
		bxPrintf("Length of label vector does not match # of instances.\n");
		fake_answer(nlhs, plhs);
		return;
	}
	if(label_vector_col_num!=1)
	{
		bxPrintf("label (1st argument) should be a vector (# of column is 1).\n");
		fake_answer(nlhs, plhs);
		return;
	}

	ptr_instance = bxGetPr(prhs[1]);
	ptr_label    = bxGetPr(prhs[0]);

	// transpose instance matrix
	if(bxIsSparse(prhs[1]))
	{
		if(model->param.kernel_type == PRECOMPUTED)
		{
			// precomputed kernel requires dense matrix, so we make one
			bxArray *rhs[1], *lhs[1];
			rhs[0] = bxDuplicateArray(prhs[1]);
			if(mexCallMATLAB(1, lhs, 1, rhs, "full"))
			{
				bxPrintf("Error: cannot full testing instance matrix\n");
				fake_answer(nlhs, plhs);
				return;
			}
			ptr_instance = bxGetPr(lhs[0]);
			bxDestroyArray(rhs[0]);
		}
		else
		{
			bxArray *pprhs[1];
			pprhs[0] = bxDuplicateArray(prhs[1]);
			if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
			{
				bxPrintf("Error: cannot transpose testing instance matrix\n");
				fake_answer(nlhs, plhs);
				return;
			}
		}
	}

	if(predict_probability)
	{
		if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
	}

	tplhs[0] = bxCreateDoubleMatrix(testing_instance_number, 1, bxREAL);
	if(predict_probability)
	{
		// prob estimates are in plhs[2]
		if(svm_type==C_SVC || svm_type==NU_SVC || svm_type==ONE_CLASS)
		{
			// nr_class = 2 for ONE_CLASS
			tplhs[2] = bxCreateDoubleMatrix(testing_instance_number, nr_class, bxREAL);
		}
		else
			tplhs[2] = bxCreateDoubleMatrix(0, 0, bxREAL);
	}
	else
	{
		// decision values are in plhs[2]
		if(svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR ||
		   nr_class == 1) // if only one class in training data, decision values are still returned.
			tplhs[2] = bxCreateDoubleMatrix(testing_instance_number, 1, bxREAL);
		else
			tplhs[2] = bxCreateDoubleMatrix(testing_instance_number, nr_class*(nr_class-1)/2, bxREAL);
	}

	ptr_predict_label = bxGetPr(tplhs[0]);
	ptr_prob_estimates = bxGetPr(tplhs[2]);
	ptr_dec_values = bxGetPr(tplhs[2]);
	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double target_label, predict_label;

		target_label = ptr_label[instance_index];

		if(bxIsSparse(prhs[1]) && model->param.kernel_type != PRECOMPUTED) // prhs[1]^T is still sparse
			read_sparse_instance(pplhs[0], instance_index, x);
		else
		{
			for(i=0;i<feature_number;i++)
			{
				x[i].index = i+1;
				x[i].value = ptr_instance[testing_instance_number*i+instance_index];
			}
			x[feature_number].index = -1;
		}

		if(predict_probability)
		{
			if(svm_type==C_SVC || svm_type==NU_SVC || svm_type==ONE_CLASS)
			{
				predict_label = svm_predict_probability(model, x, prob_estimates);
				ptr_predict_label[instance_index] = predict_label;
				for(i=0;i<nr_class;i++)
					ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
			} else {
				predict_label = svm_predict(model,x);
				ptr_predict_label[instance_index] = predict_label;
			}
		}
		else
		{
			if(svm_type == ONE_CLASS ||
			   svm_type == EPSILON_SVR ||
			   svm_type == NU_SVR)
			{
				double res;
				predict_label = svm_predict_values(model, x, &res);
				ptr_dec_values[instance_index] = res;
			}
			else
			{
				double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
				predict_label = svm_predict_values(model, x, dec_values);
				if(nr_class == 1)
					ptr_dec_values[instance_index] = 1;
				else
					for(i=0;i<(nr_class*(nr_class-1))/2;i++)
						ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
				free(dec_values);
			}
			ptr_predict_label[instance_index] = predict_label;
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);

	// return accuracy, mean squared error, squared correlation coefficient
	tplhs[1] = bxCreateDoubleMatrix(3, 1, bxREAL);
	ptr = bxGetPr(tplhs[1]);
	ptr[0] = (double)correct/total*100;
	ptr[1] = error/total;
	ptr[2] = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
				((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt));

	free(x);
	if(prob_estimates != NULL)
		free(prob_estimates);

	switch(nlhs)
	{
		case 3:
			plhs[2] = tplhs[2];
			plhs[1] = tplhs[1];
		case 1:
		case 0:
			plhs[0] = tplhs[0];
	}
}

const char* svmpredict_help;
static void exit_with_help()
{
	svmpredict_help =
		"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"       [predicted_label] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"Parameters:\n"
		"  model: SVM model structure from svmtrain.\n"
		"  libsvm_options:\n"
		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
		"    -q : quiet mode (no outputs)\n"
		"Returns:\n"
		"  predicted_label: SVM prediction output vector.\n"
		"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
		"  prob_estimates: If selected, probability estimate vector.\n"
	;
	bxPrintf(svmpredict_help);
}

void svmpredict( int nlhs, bxArray *plhs[],
		 int nrhs, const bxArray *prhs[] )
{
	int prob_estimate_flag = 0;
	struct svm_model *model;
	info = &bxPrintf;

	if(nlhs == 2 || nlhs > 3 || nrhs > 4 || nrhs < 3)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

	if(!bxIsDouble(prhs[0]) || !bxIsDouble(prhs[1])) {
		bxPrintf("Error: label vector and instance matrix must be double\n");
		fake_answer(nlhs, plhs);
		return;
	}

	if(bxIsStruct(prhs[2]))
	{
		const char *error_msg;

		// parse options
		if(nrhs==4)
		{
			int i, argc = 1;
			char cmd[CMD_LEN], *argv[CMD_LEN/2];

			// put options in argv[]
			bxAsCStr(prhs[3], cmd,  bxGetN(prhs[3]) + 1);
			if((argv[argc] = strtok(cmd, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
					;

			for(i=1;i<argc;i++)
			{
				if(argv[i][0] != '-') break;
				if((++i>=argc) && argv[i-1][1] != 'q')
				{
					exit_with_help();
					fake_answer(nlhs, plhs);
					return;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
						break;
					case 'q':
						i--;
						info = &print_null;
						break;
					default:
						bxPrintf("Unknown option: -%c\n", argv[i-1][1]);
						exit_with_help();
						fake_answer(nlhs, plhs);
						return;
				}
			}
		}

		model = matlab_matrix_to_model(prhs[2], &error_msg);
		if (model == NULL)
		{
			bxPrintf("Error: can't read model: %s\n", error_msg);
			fake_answer(nlhs, plhs);
			return;
		}

		if(prob_estimate_flag)
		{
			if(svm_check_probability_model(model)==0)
			{
				bxPrintf("Model does not support probabiliy estimates\n");
				fake_answer(nlhs, plhs);
				svm_free_and_destroy_model(&model);
				return;
			}
		}
		else
		{
			if(svm_check_probability_model(model)!=0)
				info("Model supports probability estimates, but disabled in predicton.\n");
		}

		predict(nlhs, plhs, prhs, model, prob_estimate_flag);
		// destroy model
		svm_free_and_destroy_model(&model);
	}
	else
	{
		bxPrintf("model file should be a struct array\n");
		fake_answer(nlhs, plhs);
	}

	return;
}
