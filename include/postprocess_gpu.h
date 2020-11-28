#ifndef TN_POSTPROCESS_GPU_H_
#define TN_POSTPROCESS_GPU_H_

#include<string>
#include<vector>
#include<numeric>
#include<algorithm>
#include"config.h"

int postprocessing_gpu(float* mask_pred, float*cate_pred,float* kernel_pred,
    float*&seg_preds_cpu,float*&cate_labels_cpu,float*&cate_scores_cpu);

#endif