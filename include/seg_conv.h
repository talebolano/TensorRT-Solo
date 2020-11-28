#ifndef TN_SEGCONV_H_
#define TN_SEGCONV_H_

#include<NvInfer.h>

int get_seg_preds_conv2_gpu(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks);

int get_seg_preds_conv2_cublas(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks);

#endif