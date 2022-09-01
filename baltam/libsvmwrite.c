#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libsvm.hpp"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

const char* svmwrite_help;
static void exit_with_help()
{
	svmwrite_help =
	"Usage: libsvmwrite('filename', label_vector, instance_matrix);\n"
	;
	bxPrintf(svmwrite_help);
}

static void fake_answer(int nlhs, bxArray *plhs[])
{
	int i;
	for(i=0;i<nlhs;i++)
		plhs[i] = bxCreateDoubleMatrix(0, 0, bxREAL);
}

void libsvmwrite(const char *filename, const bxArray *label_vec, const bxArray *instance_mat)
{
	FILE *fp = fopen(filename,"w");
	baIndex *ir, *jc, k, low, high;
	size_t i, l, label_vector_row_num;
	double *samples, *labels;
	bxArray *instance_mat_col; // instance sparse matrix in column format

	if(fp ==NULL)
	{
		bxPrintf("can't open output file %s\n",filename);
		return;
	}

	// transpose instance matrix
	{
#ifndef BUILD_WITH_BEX_WARPPER		
		bxArray *prhs[1], *plhs[1];
		prhs[0] = bxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			bxPrintf("Error: cannot transpose instance matrix\n");
			return;
		}
		instance_mat_col = plhs[0];
		bxDestroyArray(prhs[0]);
#else
		instance_mat_col = bxDuplicateArray(instance_mat);
#endif // BUILD_WITH_BEX_WARPPER
	}

	// the number of instance
	l = bxGetN(instance_mat_col);
	label_vector_row_num = bxGetM(label_vec);

	if(label_vector_row_num!=l)
	{
		bxPrintf("Length of label vector does not match # of instances.\n");
		return;
	}

	// each column is one instance
	labels = bxGetPr(label_vec);
	samples = bxGetPr(instance_mat_col);
	ir = bxGetIr(instance_mat_col);
	jc = bxGetJc(instance_mat_col);

	for(i=0;i<l;i++)
	{
		fprintf(fp,"%.17g", labels[i]);

		low = jc[i], high = jc[i+1];
		for(k=low;k<high;k++)
			fprintf(fp," %lu:%g", (size_t)ir[k]+1, samples[k]);

		fprintf(fp,"\n");
	}

	fclose(fp);
	return;
}

void svmwrite( int nlhs, bxArray *plhs[],
		int nrhs, const bxArray *prhs[] )
{
	if(nlhs > 0)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

	// Transform the input Matrix to libsvm format
	if(nrhs == 3)
	{
		char filename[256];
		if(!bxIsDouble(prhs[1]) || !bxIsDouble(prhs[2]))
		{
			bxPrintf("Error: label vector and instance matrix must be double\n");
			return;
		}

		bxAsCStr(prhs[0], filename, bxGetN(prhs[0])+1);

		if(bxIsSparse(prhs[2]))
			libsvmwrite(filename, prhs[1], prhs[2]);
		else
		{
			bxPrintf("Instance_matrix must be sparse\n");
			return;
		}
	}
	else
	{
		exit_with_help();
		return;
	}
}
