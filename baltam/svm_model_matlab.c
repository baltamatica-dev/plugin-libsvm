#include <stdlib.h>
#include <string.h>
#include "svm.h"

#include <bex/bex.hpp>
#include "libsvm.hpp"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define NUM_OF_RETURN_FIELD 12

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

const char *field_names[] = {
	"Parameters",
	"nr_class",
	"totalSV",
	"rho",
	"Label",
	"sv_indices",
	"ProbA",
	"ProbB",
	"Prob_density_marks",
	"nSV",
	"sv_coef",
	"SVs"
};

const char *model_to_matlab_structure(bxArray *plhs[], int num_of_feature, struct svm_model *model)
{
	int i, j, n;
	double *ptr;
	bxArray *return_model, **rhs;
	int out_id = 0;

	rhs = (bxArray **)malloc(sizeof(bxArray *)*NUM_OF_RETURN_FIELD);

	// Parameters
	rhs[out_id] = bxCreateDoubleMatrix(5, 1, bxREAL);
	ptr = bxGetPr(rhs[out_id]);
	ptr[0] = model->param.svm_type;
	ptr[1] = model->param.kernel_type;
	ptr[2] = model->param.degree;
	ptr[3] = model->param.gamma;
	ptr[4] = model->param.coef0;
	out_id++;

	// nr_class
	rhs[out_id] = bxCreateDoubleMatrix(1, 1, bxREAL);
	ptr = bxGetPr(rhs[out_id]);
	ptr[0] = model->nr_class;
	out_id++;

	// total SV
	rhs[out_id] = bxCreateDoubleMatrix(1, 1, bxREAL);
	ptr = bxGetPr(rhs[out_id]);
	ptr[0] = model->l;
	out_id++;

	// rho
	n = model->nr_class*(model->nr_class-1)/2;
	rhs[out_id] = bxCreateDoubleMatrix(n, 1, bxREAL);
	ptr = bxGetPr(rhs[out_id]);
	for(i = 0; i < n; i++)
		ptr[i] = model->rho[i];
	out_id++;

	// Label
	if(model->label)
	{
		rhs[out_id] = bxCreateDoubleMatrix(model->nr_class, 1, bxREAL);
		ptr = bxGetPr(rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->label[i];
	}
	else
		rhs[out_id] = bxCreateDoubleMatrix(0, 0, bxREAL);
	out_id++;

	// sv_indices
	if(model->sv_indices)
	{
		rhs[out_id] = bxCreateDoubleMatrix(model->l, 1, bxREAL);
		ptr = bxGetPr(rhs[out_id]);
		for(i = 0; i < model->l; i++)
			ptr[i] = model->sv_indices[i];
	}
	else
		rhs[out_id] = bxCreateDoubleMatrix(0, 0, bxREAL);
	out_id++;

	// probA
	if(model->probA != NULL)
	{
		rhs[out_id] = bxCreateDoubleMatrix(n, 1, bxREAL);
		ptr = bxGetPr(rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probA[i];
	}
	else
		rhs[out_id] = bxCreateDoubleMatrix(0, 0, bxREAL);
	out_id ++;

	// probB
	if(model->probB != NULL)
	{
		rhs[out_id] = bxCreateDoubleMatrix(n, 1, bxREAL);
		ptr = bxGetPr(rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probB[i];
	}
	else
		rhs[out_id] = bxCreateDoubleMatrix(0, 0, bxREAL);
	out_id++;

	// prob_density_marks
	if(model->prob_density_marks != NULL)
	{
		int nr_marks = 10;
		rhs[out_id] = bxCreateDoubleMatrix(nr_marks, 1, bxREAL);
		ptr = bxGetPr(rhs[out_id]);
		for(i = 0; i < nr_marks; i++)
			ptr[i] = model->prob_density_marks[i];
	}
	else
		rhs[out_id] = bxCreateDoubleMatrix(0, 0, bxREAL);
	out_id++;

	// nSV
	if(model->nSV)
	{
		rhs[out_id] = bxCreateDoubleMatrix(model->nr_class, 1, bxREAL);
		ptr = bxGetPr(rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->nSV[i];
	}
	else
		rhs[out_id] = bxCreateDoubleMatrix(0, 0, bxREAL);
	out_id++;

	// sv_coef
	rhs[out_id] = bxCreateDoubleMatrix(model->l, model->nr_class-1, bxREAL);
	ptr = bxGetPr(rhs[out_id]);
	for(i = 0; i < model->nr_class-1; i++)
		for(j = 0; j < model->l; j++)
			ptr[(i*(model->l))+j] = model->sv_coef[i][j];
	out_id++;

	// SVs
	{
		int ir_index, nonzero_element;
		baIndex *ir, *jc;
		bxArray *pprhs[1], *pplhs[1];

		if(model->param.kernel_type == PRECOMPUTED)
		{
			nonzero_element = model->l;
			num_of_feature = 1;
		}
		else
		{
			nonzero_element = 0;
			for(i = 0; i < model->l; i++) {
				j = 0;
				while(model->SV[i][j].index != -1)
				{
					nonzero_element++;
					j++;
				}
			}
		}

		// SV in column, easier accessing
		rhs[out_id] = bxCreateSparse(num_of_feature, model->l, nonzero_element, bxREAL);
		ir = bxGetIr(rhs[out_id]);
		jc = bxGetJc(rhs[out_id]);
		ptr = bxGetPr(rhs[out_id]);
		jc[0] = ir_index = 0;
		for(i = 0;i < model->l; i++)
		{
			if(model->param.kernel_type == PRECOMPUTED)
			{
				// make a (1 x model->l) matrix
				ir[ir_index] = 0;
				ptr[ir_index] = model->SV[i][0].value;
				ir_index++;
				jc[i+1] = jc[i] + 1;
			}
			else
			{
				int x_index = 0;
				while (model->SV[i][x_index].index != -1)
				{
					ir[ir_index] = model->SV[i][x_index].index - 1;
					ptr[ir_index] = model->SV[i][x_index].value;
					ir_index++, x_index++;
				}
				jc[i+1] = jc[i] + x_index;
			}
		}
		bxSparseFinalize(rhs[out_id]);

#ifndef BUILD_WITH_BEX_WARPPER
		// transpose back to SV in row
		pprhs[0] = rhs[out_id];
		if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
			return "cannot transpose SV matrix";
		rhs[out_id] = pplhs[0];
#endif // BUILD_WITH_BEX_WARPPER		
		out_id++;
	}

	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	return_model = bxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);

	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		bxSetField(return_model,0,field_names[i],bxDuplicateArray(rhs[i]));
	/* return */
	plhs[0] = return_model;
	free(rhs);

	return NULL;
}

struct svm_model *matlab_matrix_to_model(const bxArray *matlab_struct, const char **msg)
{
	int i, j, n, num_of_fields;
	double *ptr;
	int id = 0;
	struct svm_node *x_space;
	struct svm_model *model;
	bxArray **rhs;

	num_of_fields = bxGetNumberOfFields(matlab_struct);
	if(num_of_fields != NUM_OF_RETURN_FIELD)
	{
		*msg = "number of return field is not correct";
		return NULL;
	}
	rhs = (bxArray **) malloc(sizeof(bxArray *)*num_of_fields);

	for(i=0;i<num_of_fields;i++)
		rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);

	model = Malloc(struct svm_model, 1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->prob_density_marks = NULL;
	model->label = NULL;
	model->sv_indices = NULL;
	model->nSV = NULL;
	model->free_sv = 1; // XXX

	ptr = bxGetPr(rhs[id]);
	model->param.svm_type = (int)ptr[0];
	model->param.kernel_type  = (int)ptr[1];
	model->param.degree	  = (int)ptr[2];
	model->param.gamma	  = ptr[3];
	model->param.coef0	  = ptr[4];
	id++;

	ptr = bxGetPr(rhs[id]);
	model->nr_class = (int)ptr[0];
	id++;

	ptr = bxGetPr(rhs[id]);
	model->l = (int)ptr[0];
	id++;

	// rho
	n = model->nr_class * (model->nr_class-1)/2;
	model->rho = (double*) malloc(n*sizeof(double));
	ptr = bxGetPr(rhs[id]);
	for(i=0;i<n;i++)
		model->rho[i] = ptr[i];
	id++;

	// label
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->label = (int*) malloc(model->nr_class*sizeof(int));
		ptr = bxGetPr(rhs[id]);
		for(i=0;i<model->nr_class;i++)
			model->label[i] = (int)ptr[i];
	}
	id++;

	// sv_indices
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->sv_indices = (int*) malloc(model->l*sizeof(int));
		ptr = bxGetPr(rhs[id]);
		for(i=0;i<model->l;i++)
			model->sv_indices[i] = (int)ptr[i];
	}
	id++;

	// probA
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->probA = (double*) malloc(n*sizeof(double));
		ptr = bxGetPr(rhs[id]);
		for(i=0;i<n;i++)
			model->probA[i] = ptr[i];
	}
	id++;

	// probB
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->probB = (double*) malloc(n*sizeof(double));
		ptr = bxGetPr(rhs[id]);
		for(i=0;i<n;i++)
			model->probB[i] = ptr[i];
	}
	id++;

	// prob_density_marks
	if(mxIsEmpty(rhs[id]) == 0)
	{
		int nr_marks = 10;
		model->prob_density_marks = (double*) malloc(nr_marks*sizeof(double));
		ptr = bxGetPr(rhs[id]);
		for(i=0;i<nr_marks;i++)
			model->prob_density_marks[i] = ptr[i];
	}
	id++;

	// nSV
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->nSV = (int*) malloc(model->nr_class*sizeof(int));
		ptr = bxGetPr(rhs[id]);
		for(i=0;i<model->nr_class;i++)
			model->nSV[i] = (int)ptr[i];
	}
	id++;

	// sv_coef
	ptr = bxGetPr(rhs[id]);
	model->sv_coef = (double**) malloc((model->nr_class-1)*sizeof(double));
	for( i=0 ; i< model->nr_class -1 ; i++ )
		model->sv_coef[i] = (double*) malloc((model->l)*sizeof(double));
	for(i = 0; i < model->nr_class - 1; i++)
		for(j = 0; j < model->l; j++)
			model->sv_coef[i][j] = ptr[i*(model->l)+j];
	id++;

	// SV
	{
		int sr, elements;
		int num_samples;
		baIndex *ir, *jc;
		bxArray *pprhs[1], *pplhs[1];

#ifndef BUILD_WITH_BEX_WARPPER
		// transpose SV
		pprhs[0] = rhs[id];
		if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
		{
			svm_free_and_destroy_model(&model);
			*msg = "cannot transpose SV matrix";
			return NULL;
		}
		rhs[id] = pplhs[0];
#endif // BUILD_WITH_BEX_WARPPER	

		sr = (int)bxGetN(rhs[id]);

		ptr = bxGetPr(rhs[id]);
		ir = bxGetIr(rhs[id]);
		jc = bxGetJc(rhs[id]);

		num_samples = (int)bxGetNzmax(rhs[id]);

		elements = num_samples + sr;

		model->SV = (struct svm_node **) malloc(sr * sizeof(struct svm_node *));
		x_space = (struct svm_node *)malloc(elements * sizeof(struct svm_node));

		// SV is in column
		for(i=0;i<sr;i++)
		{
			int low = (int)jc[i], high = (int)jc[i+1];
			int x_index = 0;
			model->SV[i] = &x_space[low+i];
			for(j=low;j<high;j++)
			{
				model->SV[i][x_index].index = (int)ir[j] + 1;
				model->SV[i][x_index].value = ptr[j];
				x_index++;
			}
			model->SV[i][x_index].index = -1;
		}

		id++;
	}
	free(rhs);

	return model;
}
