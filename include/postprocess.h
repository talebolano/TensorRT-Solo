#ifndef TN_POSTPROCESS_H_
#define TN_POSTPROCESS_H_

#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<numeric>
#include<algorithm>
#include"config.h"
#include<chrono>


void postprocessing( cv::Mat &img,std::string &filename,float* mask_pred, float*cate_pred,float* kernel_pred);
void vis_seg(cv::Mat &img,float*seg_preds,float*cate_labels,float*cate_scores,int ind_size,
            std::string filename,bool write,bool show=false);
void vis_seg(cv::VideoWriter &videowriter,cv::Mat &img,float*seg_preds,float*cate_labels,float*cate_scores,int ind_size,
            std::string filename,bool write,bool show=false);

#endif

    

