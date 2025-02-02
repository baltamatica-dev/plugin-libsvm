/* SPDX-License-Identifier: BSD-3-Clause */
#include "libsvm.hpp"


/* workaround */

bxArray *mxGetFieldByNumber(const bxArray *pm, baIndex index, int fieldnumber) {
    return bxGetField(pm, index, field_names[fieldnumber]);
}

bool mxIsEmpty(const bxArray *arr) {
    return (0==bxGetM(arr)) || (0==bxGetN(arr));
}

int mexCallMATLAB(int nlhs, bxArray *plhs[], int nrhs, bxArray *prhs[], const char *functionName) {
	return 1;
}


const char* svm_version_help = R"(
libsvm 绑定

    libsvm 插件 (v20220903+12:00-alpha-测试版)
    Github:     https://github.com/baltamatica-dev/plugin-libsvm
    LICENSE:    BSD-3-Clause license
    Copyright (c) 2022 Chengyu HAN

版本信息
    bex SDK: 2.2.1
    cjlin1/libsvm (324612f)
        Github:     https://github.com/cjlin1/libsvm
        LICENSE:    BSD-3-Clause license
        Copyright (c) Chih-Chung Chang and Chih-Jen Lin
)"; /* svm_version_help */
BALTAM_PLUGIN_FCN(svm_version) {
    bxPrintf(svm_version_help);
} /* svm_version */


/**
 * @brief [可选] 插件初始化函数.
 *
 * bxPluginInit 由 load_plugin(name, args...) 调用
 * 用于进行一些初始化工作
 *
 * @param nInit
 * @param pInit[]
 */
int bxPluginInit(int nInit, const bxArray* pInit[]) {
    return 0;
} /* bxPluginInit */

/**
 * @brief [可选] 插件终止时清理函数.
 *
 * bxPluginFini 由 unload_plugin() 调用
 * 用于进行一些清理工作
 */
int bxPluginFini() {
    return 0;
} /* bxPluginFini */

/**
 * @brief 【必选】 列出插件提供的函数.
 *
 * bxPluginFunctions 返回 指向函数列表的指针.
 */
bexfun_info_t * bxPluginFunctions() {
    // 已定义的插件函数个数
    constexpr size_t TOTAL_PLUGIN_FUNCTIONS = 5;
    bexfun_info_t* func_list_dyn = new bexfun_info_t[TOTAL_PLUGIN_FUNCTIONS + 1];

    size_t i = 0;
    func_list_dyn[i].name = "svm_version";
    func_list_dyn[i].ptr  = svm_version;
    func_list_dyn[i].help = svm_version_help;

    i++;
    func_list_dyn[i].name = "__svmpredict_impl";
    func_list_dyn[i].ptr  = svmpredict;
    func_list_dyn[i].help = "See help for `svmpredict`\n  You should call `svminit();` first.";

    i++;
    func_list_dyn[i].name = "__svmtrain_impl";
    func_list_dyn[i].ptr  = svmtrain;
    func_list_dyn[i].help = "See help for `svmtrain`\n  You should call `svminit();` first.";

    i++;
    func_list_dyn[i].name = "__svmread_impl";
    func_list_dyn[i].ptr  = svmread;
    func_list_dyn[i].help = "See help for `svmread`\n  You should call `svminit();` first.";
    
    i++;
    func_list_dyn[i].name = "__svmwrite_impl";
    func_list_dyn[i].ptr  = svmwrite;
    func_list_dyn[i].help = "See help for `svmwrite`\n  You should call `svminit();` first.";

    // 最后一个元素, `name` 字段必须为空字符串 `""`
    i++;
    func_list_dyn[i].name = "";
    func_list_dyn[i].ptr  = nullptr;
    func_list_dyn[i].help = nullptr;

    assert((TOTAL_PLUGIN_FUNCTIONS == i));
    return func_list_dyn;
} /* bxPluginFunctions */
