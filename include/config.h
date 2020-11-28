#ifndef __CONFIG_H_
#define __CONFIG_H_

#include<numeric>
#include<string>
template<class T>
int getArraylen(T& array){
    return sizeof(array)/sizeof(array[0]);
}

const int seg_num_grids[5] = {40, 36, 24, 16, 12};
const int self_strides[5] = {8, 8, 16, 32, 32};
const float mask_thr = 0.5;
const int max_per_img = 100;
const float vis_thresh=0.3;

const int outputsize[2] = {1024,720};
const int cate_pred_shape[2] ={3872,80};//如果改变类别，在这里更改
const int kernel_pred_shape[2] ={3872,256};
const int mask_pred_shape[3]={256,200,336};

const int cate_pred_num = getArraylen(cate_pred_shape);
const int kernel_pred_num = getArraylen(kernel_pred_shape);
const int mask_pred_num = getArraylen(mask_pred_shape);

const int cate_pred_size =std::accumulate(cate_pred_shape,cate_pred_shape+cate_pred_num,1,std::multiplies<int64_t>());
const int kernel_pred_size =std::accumulate(kernel_pred_shape,kernel_pred_shape+kernel_pred_num,1,std::multiplies<int64_t>());
const int mask_pred_size =std::accumulate(mask_pred_shape,mask_pred_shape+mask_pred_num,1,std::multiplies<int64_t>());

//如果改变类别，在这里更改
const std::string class_names[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
               "train", "truck", "boat", "traffic_light", "fire_hydrant",
               "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
               "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports_ball", "kite", "baseball_bat",
               "baseball_glove", "skateboard", "surfboard", "tennis_racket",
               "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
               "hot_dog", "pizza", "donut", "cake", "chair", "couch",
               "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
               "mouse", "remote", "keyboard", "cell_phone", "microwave",
               "oven", "toaster", "sink", "refrigerator", "book", "clock",
               "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"};


#endif