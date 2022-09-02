#pragma once
#include <iostream>
#include <cassert>
#include <bex/bex.hpp>

#define BUILD_WITH_BEX_WARPPER


extern const char *model_to_matlab_structure(bxArray *plhs[], int num_of_feature, struct svm_model *model);
extern struct svm_model *matlab_matrix_to_model(const bxArray *matlab_struct, const char **error_message);
extern const char *field_names[]; 


BALTAM_PLUGIN_FCN(svmpredict);
// extern const char* svmpredict_help;
BALTAM_PLUGIN_FCN(svmtrain);
// extern const char* svmtrain_help;
BALTAM_PLUGIN_FCN(svmread);
// extern const char* svmread_help;
BALTAM_PLUGIN_FCN(svmwrite);
// extern const char* svmwrite_help;

extern bxArray *mxGetFieldByNumber(const bxArray *pm, baIndex index, int fieldnumber);
extern bool mxIsEmpty(const bxArray *arr);
extern int mexCallMATLAB(int nlhs, bxArray *plhs[], int nrhs, bxArray *prhs[], const char *functionName);
