#include"postprocess.h"
#include<random>
#include "seg_conv.h"

int get_cate_ind(float* cate_perds,float* kernel_pred,float *cate_scores,float*cate_label,
                    float*kernel);

void get_seg_preds_conv2(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks);

void get_cate_scores(float*seg_preds,float*seg_masks,float*sum_masks,float*cate_scores,
                    int ind_size);

void argsort_all_need(float *cate_scores,float*seg_masks,float*seg_preds,float*sum_masks,
            float*cate_labels,int& ind_size,
            float *&sort_cate_scores,float*&sort_cate_labels,float*&sort_sum_masks,
            float*&sort_seg_preds,float*&sort_seg_masks);

int* argsort(float*cate_scores,int& ind_size);

void matrix_nms(float*seg_masks,float*cate_labels, float*cate_scores,float*sum_masks,int ind_size);


cv::Scalar random_color();

cv::Point getCenterPoint(cv::Rect rect);

void postprocessing( cv::Mat &img,std::string &filename,float* mask_pred, float*cate_pred,float* kernel_pred){
    float *cate_scores;
    cate_scores = (float*)calloc(cate_pred_size,sizeof(float));

    float*cate_label;
    cate_label = (float*)calloc(cate_pred_size,sizeof(float));

    float*kernel;
    kernel = (float*)calloc(kernel_pred_size,sizeof(float));
    int ind_size = get_cate_ind(cate_pred,kernel_pred,cate_scores,cate_label,kernel);

    if (ind_size==0){

        free(cate_scores);
        free(cate_label);
        free(kernel);
        cv::imwrite(filename,img);
        return;
    }

    float*seg_preds_dst;
    seg_preds_dst = (float*)calloc(ind_size*mask_pred_shape[1]*mask_pred_shape[2],sizeof(float));

    float*sum_masks;
    sum_masks = (float*)calloc(ind_size,sizeof(float));

    float*seg_masks;
    seg_masks = (float*)calloc(ind_size*mask_pred_shape[1]*mask_pred_shape[2],sizeof(float));
    get_seg_preds_conv2_gpu(mask_pred,kernel,ind_size,seg_preds_dst,sum_masks,seg_masks);

    get_cate_scores(seg_preds_dst,seg_masks,sum_masks,cate_scores,ind_size);

    float*sort_cate_scores;
    float*sort_cate_labels;
    float*sort_sum_masks;
    float*sort_seg_preds;
    float*sort_seg_masks;

    argsort_all_need(cate_scores,seg_masks,seg_preds_dst,sum_masks,cate_label,
            ind_size,
            sort_cate_scores,sort_cate_labels,sort_sum_masks,
            sort_seg_preds,sort_seg_masks);

    free(cate_scores);
    free(seg_masks);
    free(seg_preds_dst);
    free(sum_masks);
    free(cate_label);
    free(kernel);

    matrix_nms(sort_seg_masks,sort_cate_labels,sort_cate_scores,sort_sum_masks,ind_size);

    free(sort_seg_masks);
    free(sort_sum_masks);
    vis_seg(img,sort_seg_preds,sort_cate_labels,sort_cate_scores,ind_size,filename,true);
    free(sort_seg_preds);
    free(sort_cate_labels);
    free(sort_cate_scores);

}


int get_cate_ind(float* cate_perds,float* kernel_pred,float *cate_scores,float*cate_label,
                    float*kernel){

    int cate_pred_shape_0 = cate_pred_shape[0];
    int cate_pred_shape_1 = cate_pred_shape[1];
    int ind_size = 0;

    for(int i=0;i<cate_pred_shape_0;++i){  
        for(int j=0;j<cate_pred_shape_1;++j)
        {
            if(cate_perds[i*cate_pred_shape_1+j]>0.1){                
                cate_scores[ind_size] = cate_perds[i*cate_pred_shape_1+j];

                cate_label[ind_size] = j;
                for(int z = 0;z<kernel_pred_shape[1];++z){ 
                    
                    kernel[ind_size*kernel_pred_shape[1]+z] = kernel_pred[i*kernel_pred_shape[1]+z];
                }

                ind_size++;
                
            }
            
        }
    }
    return ind_size;

}


void get_seg_preds_conv2(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks){

    int out_H_W = mask_pred_shape[1]*mask_pred_shape[2];
    for(int ind=0;ind<ind_size;++ind)
    {
        for(int h_w=0;h_w<out_H_W;++h_w){ 
            
            for(int n=0;n<mask_pred_shape[0];++n){

                dst[ind*out_H_W+h_w] +=kernels[ind*mask_pred_shape[0]+n]*seg_preds[n*out_H_W+h_w];
            
            }
            dst[ind*out_H_W+h_w] = 1./(1.+exp(-dst[ind*out_H_W+h_w]));

            if (dst[ind*out_H_W+h_w]>0.5){
                sum_masks[ind] +=1.;
                seg_masks[ind*out_H_W+h_w] =1.; 

            }
        }
    }

return;

}


void get_cate_scores(float*seg_preds,float*seg_masks,float*sum_masks,float*cate_scores,
                    int ind_size){

    float*temp;
    temp = (float*)calloc(ind_size,sizeof(float));

    int out_H_W = mask_pred_shape[1]*mask_pred_shape[2];
    for(int ind=0;ind<ind_size;++ind){
        for(int h_w=0;h_w<out_H_W;++h_w){

            temp[ind]+= seg_preds[ind*out_H_W+h_w]*seg_masks[ind*out_H_W+h_w]/sum_masks[ind];
        }
        cate_scores[ind] *= temp[ind];

    }
    free(temp);

}

void argsort_all_need(float *cate_scores,float*seg_masks,float*seg_preds,float*sum_masks,
            float*cate_labels,int& ind_size,
            float *&sort_cate_scores,float*&sort_cate_labels,float*&sort_sum_masks,
            float*&sort_seg_preds,float*&sort_seg_masks){
    
    int *cate_score_index;
    cate_score_index = argsort(cate_scores,ind_size);
    std::cout<<"ind size is"<<ind_size<<std::endl;

    if (ind_size>max_per_img){
        ind_size = max_per_img;
    }

    sort_cate_scores = (float*)calloc(ind_size,sizeof(float));
    sort_cate_labels = (float*)calloc(ind_size,sizeof(float));
    sort_sum_masks = (float*)calloc(ind_size,sizeof(float));
    sort_seg_preds = (float*)calloc(ind_size*mask_pred_shape[1]*mask_pred_shape[2],sizeof(float));
    sort_seg_masks = (float*)calloc(ind_size*mask_pred_shape[1]*mask_pred_shape[2],sizeof(float));

    int out_H_W = mask_pred_shape[1]*mask_pred_shape[2];
    for(int ind=0;ind<ind_size;++ind){

        sort_cate_labels[ind] = cate_labels[cate_score_index[ind]];
        sort_cate_scores[ind] = cate_scores[cate_score_index[ind]];
        sort_sum_masks[ind] = sum_masks[cate_score_index[ind]];

        for(int h_w=0;h_w<out_H_W;++h_w){
            sort_seg_masks[ind*out_H_W+h_w] = seg_masks[cate_score_index[ind]*out_H_W+h_w];

            sort_seg_preds[ind*out_H_W+h_w] = seg_preds[cate_score_index[ind]*out_H_W+h_w];

        }
    }
    free(cate_score_index);

}


int* argsort(float*cate_scores,int& ind_size){

    int*temp;
    temp = (int*)calloc(ind_size,sizeof(int));
    for(int ind = 0;ind<ind_size;++ind){
        temp[ind] = ind;
    }

    std::sort(temp,temp+ind_size,
            [&cate_scores](int pos1,int pos2){return (cate_scores[pos1]>cate_scores[pos2]);});
    return temp;
}


void matrix_nms(float*seg_masks,float*cate_labels, float*cate_scores,float*sum_masks,int ind_size){
    
    int out_H_W = mask_pred_shape[1]*mask_pred_shape[2];

    float*inter_matrix;
    inter_matrix = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*iou_matrix;
    iou_matrix = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*label_matrix;
    label_matrix = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*compensate_iou;
    compensate_iou = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*decay_iou;
    decay_iou = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*decay_matrix;
    decay_matrix = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*compensate_matrix;
    compensate_matrix = (float*)calloc(ind_size*ind_size,sizeof(float));

    float*decay_coefficient;
    decay_coefficient = (float*)calloc(ind_size,sizeof(float));


    for(int h=0;h<ind_size;++h){
        for(int w=0;w<ind_size;++w){
            for(int h_w=0;h_w<out_H_W;++h_w){
                inter_matrix[h*ind_size+w] += seg_masks[h*out_H_W+h_w]*seg_masks[w*out_H_W+h_w];
            }
        }
    }


    for(int h=0;h<ind_size;++h){
        for(int w=0;w<ind_size;++w){

            iou_matrix[h*ind_size+w] = inter_matrix[h*ind_size+w]/(sum_masks[w]+sum_masks[h]-inter_matrix[h*ind_size+w]);

        }
        for(int w=0;w<h+1;++w){
            iou_matrix[h*ind_size+w] = 0;
        }
    }

    for(int h=0;h<ind_size;++h){
        for(int w=0;w<ind_size;++w){
            if(cate_labels[w]==cate_labels[h]){

                label_matrix[h*ind_size+w] = 1.;
            }
        }
        for(int w=0;w<h+1;++w){
            label_matrix[h*ind_size+w] = 0;
            
        }

    }

    float*temp_max;
    temp_max = (float*)calloc(ind_size*ind_size,sizeof(float));
    for(int h=0;h<ind_size;++h){
        for(int w=0;w<ind_size;++w){
            decay_iou[h*ind_size+w] = iou_matrix[h*ind_size+w]*label_matrix[h*ind_size+w];
            temp_max[w*ind_size+h] = decay_iou[h*ind_size+w];

        }
        
    }

    for(int h=0;h<ind_size;++h){
        for(int w=0;w<ind_size;++w){
            compensate_iou[h*ind_size+w] = *std::max_element(temp_max+h*ind_size,temp_max+h*ind_size+ind_size);;
        }
    }


    float*temp_min;
    temp_min = (float*)calloc(ind_size*ind_size,sizeof(float));

    for(int h=0;h<ind_size;++h){
        for(int w=0;w<ind_size;++w){
            compensate_matrix[h*ind_size+w] = exp(-2.* pow(compensate_iou[h*ind_size+w],2));
            decay_matrix[h*ind_size+w] =  exp(-2.* pow(decay_iou[h*ind_size+w],2)) / compensate_matrix[h*ind_size+w];
            temp_min[w*ind_size+h] = decay_matrix[h*ind_size+w];

        }

    }
    
    for(int h=0;h<ind_size;++h){
        decay_coefficient[h] = *std::min_element(temp_min+h*ind_size,temp_min+h*ind_size+ind_size);
    
    }
    for(int i=0;i<ind_size;++i){
        cate_scores[i] *=decay_coefficient[i];
    }

    free(temp_max);
    free(temp_min);
    free(inter_matrix);
    free(iou_matrix);
    free(label_matrix);
    free(compensate_iou);
    free(decay_iou);
    free(decay_matrix);
    free(compensate_matrix);
    free(decay_coefficient);


}



void vis_seg(cv::Mat &img,float*seg_preds,float*cate_labels,float*cate_scores,int ind_size,
            std::string filename,bool write,bool show){
    if(ind_size==0){

        if(write){
            cv::imwrite(filename,img);
        }
        if(show){
            cv::imshow("output image",img);
            cv::waitKey(10);
        }
        
        return;
    }

    cv::Mat draw_image = img.clone();
    for(int i=0;i<ind_size;++i){
        if (cate_scores[i]>vis_thresh){
            cv::Mat mask_pred(mask_pred_shape[1],mask_pred_shape[2],CV_32FC1);    
            for(int h=0;h<mask_pred_shape[1];++h){
                cv::Vec<float,1>*p1 = mask_pred.ptr<cv::Vec<float,1>>(h);
                for(int w=0;w<mask_pred_shape[2];++w){
                    p1[w][0] = seg_preds[i*mask_pred_shape[1]*mask_pred_shape[2]+h*mask_pred_shape[2]+w];

                }
            }
            cv::Mat reiszed(outputsize[1],outputsize[0],CV_32FC1);

            cv::resize(mask_pred,reiszed,reiszed.size());

            cv::Mat mask(outputsize[1],outputsize[0],CV_8U,cv::Scalar(0));

            reiszed.convertTo(mask,CV_8U);//

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask,contours,cv::noArray(),cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
            cv::Scalar random_scalar = random_color();
            cv::drawContours(draw_image,contours,-1,random_scalar,5);
            int cur_label = (int) cate_labels[i];
            std::string label_txt = class_names[cur_label];

            cv::Rect rect = cv::boundingRect(mask);
            cv::putText(draw_image,label_txt,getCenterPoint(rect),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(255,255,255),2);

        }
    
    }
    if(write){
        cv::imwrite(filename,draw_image);
    }
    if(show){
        
        cv::imshow("output image",draw_image);
        cv::waitKey(10);
    }

    return;
}


void vis_seg(cv::VideoWriter &videowriter,cv::Mat &img,float*seg_preds,float*cate_labels,float*cate_scores,int ind_size,
            std::string filename,bool write,bool show){
    if(ind_size==0){

        if(write){
            videowriter.write(img);
        }
        if(show){
            cv::imshow("output image",img);
            cv::waitKey(10);
        }
        
        return;
    }

    cv::Mat draw_image = img.clone();
    for(int i=0;i<ind_size;++i){
        if (cate_scores[i]>vis_thresh){
            cv::Mat mask_pred(mask_pred_shape[1],mask_pred_shape[2],CV_32FC1);    
            for(int h=0;h<mask_pred_shape[1];++h){
                cv::Vec<float,1>*p1 = mask_pred.ptr<cv::Vec<float,1>>(h);
                for(int w=0;w<mask_pred_shape[2];++w){
                    p1[w][0] = seg_preds[i*mask_pred_shape[1]*mask_pred_shape[2]+h*mask_pred_shape[2]+w];

                }
            }
            cv::Mat reiszed(outputsize[1],outputsize[0],CV_32FC1);
            cv::resize(mask_pred,reiszed,reiszed.size());
            cv::Mat mask(outputsize[1],outputsize[0],CV_8U,cv::Scalar(0));
            reiszed.convertTo(mask,CV_8U);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask,contours,cv::noArray(),cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
            cv::Scalar random_scalar = random_color();
            cv::drawContours(draw_image,contours,-1,random_scalar,5);
            int cur_label = (int) cate_labels[i];
            std::string label_txt = class_names[cur_label];
            cv::Rect rect = cv::boundingRect(mask);
            cv::putText(draw_image,label_txt,getCenterPoint(rect),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(255,255,255),2);

        }
    

    }
    if(write){
        videowriter.write(draw_image);
    }
    if(show){
        
        cv::imshow("output image",draw_image);
        cv::waitKey(10);
    }

    return;
}

cv::Scalar random_color(){

    std::random_device rd;
    int r = rd()%256;
    int g = rd()%256;
    int b = rd()%256;

    return cv::Scalar(r,g,b);

}

cv::Point getCenterPoint(cv::Rect rect){
    cv::Point cxy;
    cxy.x = rect.x+cvRound(rect.width/2.);
    cxy.y = rect.y+cvRound(rect.height/2.);
    return cxy;

}