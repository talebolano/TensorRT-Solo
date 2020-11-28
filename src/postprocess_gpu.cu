#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cublas_v2.h"
#include "device_atomic_functions.h"
#include "config.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <float.h>
#include <chrono>

#define BLOCK_DIM 32


struct is_morethan_thresh
{
    __host__ __device__
    bool operator()(float &x){
        return x>0.3; 
    }
};

__global__ void get_cate_ind_kernel(float*cate_pred,float* kernel_pred,
                                    int kernel_shape,int cate_pred_shape_1,
                                    int ind_size,int*g_temp_index,
                                    float *cate_scores,float*cate_label,float*kernel){
    
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int h_w = blockIdx.y * blockDim.y + threadIdx.y;
    if (ind>ind_size-1 ) return;
    else{
        if (h_w<kernel_shape) {

            cate_scores[ind] = cate_pred[ind];

            cate_label[ind] = g_temp_index[ind]%cate_pred_shape_1;

            kernel[ind*kernel_shape+h_w] = kernel_pred[g_temp_index[ind]/cate_pred_shape_1*kernel_shape+h_w];

        }

    }

}


int get_cate_ind_gpu(float* cate_perds,float* kernel_pred,
                    float *cate_scores,float*cate_label,
                    float*kernel){

    const int cate_pred_shape_0 = cate_pred_shape[0];
    const int cate_pred_shape_1 = cate_pred_shape[1];
    const int kernel_shape = kernel_pred_shape[1];

    int ind_size;
    ind_size = thrust::count_if(thrust::device,cate_perds,cate_perds+cate_pred_shape[0]*cate_pred_shape[1],is_morethan_thresh());
    cudaDeviceSynchronize();

    int*g_temp_index;
    cudaMalloc((void**)&g_temp_index,cate_pred_shape[0]*cate_pred_shape[1]*sizeof(int));

    thrust::sequence(thrust::device,g_temp_index,g_temp_index+cate_pred_shape[0]*cate_pred_shape[1]);
    cudaDeviceSynchronize();
    thrust::stable_sort_by_key(thrust::device,cate_perds,cate_perds+cate_pred_shape[0]*cate_pred_shape[1],g_temp_index,thrust::greater<float>());
    cudaDeviceSynchronize();
    dim3 mygird((ind_size+BLOCK_DIM-1)/BLOCK_DIM,(kernel_shape+BLOCK_DIM-1)/BLOCK_DIM);
    
    dim3 myBlock(BLOCK_DIM,BLOCK_DIM);
    get_cate_ind_kernel<<<mygird,myBlock>>>(cate_perds,kernel_pred,
                                            kernel_shape,cate_pred_shape_1,
                                            ind_size,g_temp_index,
                                            cate_scores,cate_label,kernel);

    cudaDeviceSynchronize();
    cudaFree(g_temp_index);
    cudaDeviceSynchronize();

    return ind_size;

}

__device__ float sigmoid(float src){
    
    return 1./(1.+expf(-src));

}

__global__ void get_conv_kernel(float*kernel,float*src,float* dst,float*seg_masks,int hA,int wA,int wB){
    

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Csub = 0.0f;

    for(int j=0;j<wA;j+=BLOCK_DIM){
        __shared__ float AS[BLOCK_DIM][BLOCK_DIM];
        __shared__ float BS[BLOCK_DIM][BLOCK_DIM];

        if(((by*BLOCK_DIM+ty)<hA)&&((tx+j)<wA)){
            AS[ty][tx]=kernel[(by*BLOCK_DIM+ty)*wA+tx+j];
        }
        else{
            AS[ty][tx]=0;
        }
        if(((ty+j)<wA)&&((bx*BLOCK_DIM+tx)<wB)){
            BS[ty][tx] = src[(ty+j)*wB+bx*BLOCK_DIM+tx];
        }
        else{
            BS[ty][tx]=0;
        }

        __syncthreads();

        for(int k=0;k<BLOCK_DIM;++k){
            Csub+=AS[ty][k]*BS[k][tx];
        }

        __syncthreads();
        
    }

    if(((by*BLOCK_DIM+ty)<hA)&&((bx*BLOCK_DIM+tx)<wB)){
        int c = wB*BLOCK_DIM*by+BLOCK_DIM*bx;
        dst[c+wB*ty+tx] = Csub;
        dst[c+wB*ty+tx] = sigmoid(dst[c+wB*ty+tx]);
        if(dst[c+wB*ty+tx]>0.5){
            seg_masks[c+wB*ty+tx] = 1.;


        }
        else{
            seg_masks[c+wB*ty+tx] = 0.;
        }
        
    }


}

__global__ void get_sum_mask_kernel(float*seg_masks,float*sum_masks,int ind_size,int w_h){

    __shared__ float cache[1024];

    int cacheindex = threadIdx.x;
    float temp=0;
    int tid_y = blockIdx.x;
    for(int tid_x=threadIdx.x;tid_x<w_h;tid_x+=blockDim.x){
        temp+=seg_masks[tid_y*w_h+tid_x];
    }
    cache[cacheindex] = temp;
    __syncthreads();

    int i =blockDim.x/2;
    while (i!=0)
    {
        if(cacheindex<i){
            cache[cacheindex]+=cache[cacheindex+i];
        }
        __syncthreads();
        i/=2;
    }
    if(cacheindex==0){
        sum_masks[tid_y] = cache[0];
    }
    
}


int get_seg_preds_conv_gpu(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks){

    const int h_w = mask_pred_shape[1]*mask_pred_shape[2];
    const int wB = h_w;
    const int hA = ind_size;

    dim3 mygird((wB+BLOCK_DIM-1)/BLOCK_DIM,(hA+BLOCK_DIM-1)/BLOCK_DIM);
    dim3 myBlock(BLOCK_DIM,BLOCK_DIM);
    get_conv_kernel<<<mygird,myBlock>>>(kernels,seg_preds,dst,seg_masks,hA,kernel_pred_shape[1],wB);
    cudaDeviceSynchronize();
    get_sum_mask_kernel<<<ind_size,1024>>>(seg_masks,sum_masks,ind_size,h_w);

    cudaDeviceSynchronize();

    return 0;
}


__global__ void get_cate_scores_kernel(float*temp_scores,float*sum_masks,float*cate_scores,
                    int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;

    float tmp=0.0f;

    if(sum_masks[idx]>0){

        tmp = temp_scores[idx]/sum_masks[idx];
        cate_scores[idx] *= tmp;
    }
    else
    {
        cate_scores[idx]=0.;
    }
    
}

__global__ void get_temp_scores_kernel(float*seg_preds,float*seg_masks,float*temp_scores,
                    int ind_size,int out_H_W){

    __shared__ float cache[1024];

    int cacheindex = threadIdx.x;
    float temp=0;
    int tid_y = blockIdx.x;
    for(int tid_x=threadIdx.x;tid_x<out_H_W;tid_x+=blockDim.x){
        temp+=seg_masks[tid_y*out_H_W+tid_x] * seg_preds[tid_y*out_H_W+tid_x];
    }
    cache[cacheindex] = temp;
    __syncthreads();

    int i =blockDim.x/2;
    while (i!=0)
    {
        if(cacheindex<i){
            cache[cacheindex]+=cache[cacheindex+i];
        }
        __syncthreads();
        i/=2;
    }
    if(cacheindex==0){
        temp_scores[tid_y] = cache[0];
    }
    
}


void get_cate_scores_gpu(float*seg_preds,float*seg_masks,float*sum_masks,float*cate_scores,
                    int ind_size){
    int out_H_W = mask_pred_shape[1]*mask_pred_shape[2];
    float*temp_scores;
    cudaMalloc((void**)&temp_scores,ind_size*sizeof(float));

    get_temp_scores_kernel<<<ind_size,1024>>>(seg_preds,seg_masks,temp_scores,ind_size,out_H_W);
    cudaDeviceSynchronize();
    const int blockSize = 512;
    const int gridSize = (ind_size + blockSize - 1) / blockSize;
    get_cate_scores_kernel<<<gridSize,blockSize>>>(temp_scores,sum_masks,cate_scores,ind_size);

    cudaDeviceSynchronize();

    cudaFree(temp_scores);
    return;


}


__global__ void get_sorted_all_need_kernel(float*cate_labels,float*sum_masks,float*seg_masks,float*seg_preds,
            int ind_size,int out_H_W,int*g_temp_index,
            float*sort_cate_labels,float*sort_sum_masks,
            float*sort_seg_preds,float*sort_seg_masks){
    
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int h_w = blockIdx.y * blockDim.y + threadIdx.y;
    if (ind>ind_size-1 ) return;
    else{
        if (h_w<out_H_W) {
            sort_cate_labels[ind] = cate_labels[g_temp_index[ind]];

            sort_sum_masks[ind] = sum_masks[g_temp_index[ind]];

            sort_seg_masks[ind*out_H_W+h_w] = seg_masks[g_temp_index[ind]*out_H_W+h_w];
            sort_seg_preds[ind*out_H_W+h_w] = seg_preds[g_temp_index[ind]*out_H_W+h_w];
        }

    }

}


void argsort_all_need_gpu(float*cate_scores,float*seg_masks,float*seg_preds,float*sum_masks,
            float*cate_labels,int& ind_size,
            float*&sort_cate_labels,float*&sort_sum_masks,
            float*&sort_seg_preds,float*&sort_seg_masks){

    
    int out_H_W = mask_pred_shape[1]*mask_pred_shape[2];

    int*g_temp_index;

    cudaMalloc((void**)&g_temp_index,ind_size*sizeof(int));

    thrust::sequence(thrust::device,g_temp_index,g_temp_index+ind_size);

    cudaDeviceSynchronize();

    thrust::stable_sort_by_key(thrust::device,cate_scores,cate_scores+ind_size,g_temp_index,thrust::greater<float>());
    cudaDeviceSynchronize();

    ind_size = thrust::count_if(thrust::device,cate_scores,cate_scores+ind_size,is_morethan_thresh());
    cudaDeviceSynchronize();
    if (ind_size>max_per_img){
        ind_size=max_per_img;
    }

    //cudaDeviceSynchronize();
    cudaMalloc((void**)&sort_cate_labels,ind_size*sizeof(float));
    cudaMalloc((void**)&sort_sum_masks,ind_size*sizeof(float));
    cudaMalloc((void**)&sort_seg_preds,ind_size*mask_pred_shape[1]*mask_pred_shape[2]*sizeof(float));
    cudaMalloc((void**)&sort_seg_masks,ind_size*mask_pred_shape[1]*mask_pred_shape[2]*sizeof(float));


    dim3 mygird((ind_size+BLOCK_DIM-1)/BLOCK_DIM,(out_H_W+BLOCK_DIM-1)/BLOCK_DIM);    
    dim3 myBlock(BLOCK_DIM,BLOCK_DIM);
    get_sorted_all_need_kernel<<<mygird,myBlock>>>(cate_labels,sum_masks,seg_masks,seg_preds,ind_size,out_H_W,g_temp_index,sort_cate_labels,sort_sum_masks,sort_seg_preds,sort_seg_masks);
    
    cudaFree(g_temp_index);

    cudaDeviceSynchronize();
}

__global__ void get_inter_matrix_kernel(float*seg_masks,float*inter_matrix,int hA,int wA,int wB){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Csub = 0.0f;

    for(int j=0;j<wA;j+=BLOCK_DIM){
        __shared__ float AS[BLOCK_DIM][BLOCK_DIM];
        __shared__ float BS[BLOCK_DIM][BLOCK_DIM];

        if(((by*BLOCK_DIM+ty)<hA)&&((tx+j)<wA)){
            AS[ty][tx]=seg_masks[(by*BLOCK_DIM+ty)*wA+tx+j];
            //BS[ty][tx]=seg_masks[(by*BLOCK_DIM+ty)*wA+tx+j];
        }
        else{
            AS[ty][tx]=0;
            BS[ty][tx]=0;
        }
        if(((ty+j)<wA)&&((bx*BLOCK_DIM+tx)<wB)){
            BS[ty][tx] = seg_masks[(ty+j)+(bx*BLOCK_DIM+tx)*wA]; //seg_masks[(ty+j)*wB+bx*BLOCK_DIM+tx];
        }
        else{
            BS[ty][tx]=0;
        }

        __syncthreads();

        for(int k=0;k<BLOCK_DIM;++k){
            Csub+=AS[ty][k]*BS[k][tx];
        }

        __syncthreads();
        
    }

    if(((by*BLOCK_DIM+ty)<hA)&&((bx*BLOCK_DIM+tx)<wB)){
        int c = wB*BLOCK_DIM*by+BLOCK_DIM*bx;
        inter_matrix[c+wB*ty+tx] = Csub;
        
    }

}


__global__ void get_iou_matrix_kernel(float*inter_matrix,float*sum_masks,float*iou_matrix,int ind_size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;

    for(int i=0;i<ind_size;++i){
        if(i>idx){
            iou_matrix[idx*ind_size+i] = inter_matrix[idx*ind_size+i]/(sum_masks[i]+sum_masks[idx]-inter_matrix[idx*ind_size+i]);
        }
        else
        {
            iou_matrix[idx*ind_size+i] = 0.;
        }
    }

}


__global__ void get_label_matrix_kernel(float*cate_labels,float*label_matrix,int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;
    for(int i=0;i<ind_size;++i){
        if(i>idx){
            if(cate_labels[idx]==cate_labels[i]){
                label_matrix[idx*ind_size+i] = 1.;
            }
        }
        else
        {
            label_matrix[idx*ind_size+i] = 0.;
        }
    }

}


__global__ void get_decay_iou_kernel(float*inter_matrix,float*label_matrix,float*decay_iou,int ind_size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;
    for(int i=0;i<ind_size;++i){
        decay_iou[idx*ind_size+i] = inter_matrix[idx*ind_size+i]*label_matrix[idx*ind_size+i];
    }
}

__global__ void get_max_kernel(float*decay_iou,float*temp_max,int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;
    float max = 0.;
    for(int i=0;i<ind_size;++i){
        if(decay_iou[i*ind_size+idx]>max){
            max = decay_iou[i*ind_size+idx];
        }
    }
    temp_max[idx] = max;

}

__global__ void get_compensate_iou_kernel(float*temp_max,float*compensate_iou,int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;
    for(int i=0;i<ind_size;++i){

        compensate_iou[idx*ind_size+i] = temp_max[idx];
    }


}

__global__ void get_decay_compensate_maxtrix_kernel(float*decay_iou,float*compensate_iou,
                                        float*decay_matrix,float*compensate_matrix,int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;

    decay_matrix[idx] = expf(-2.*powf(decay_iou[idx],2));
    compensate_matrix[idx] = expf(-2.*powf(compensate_iou[idx],2));

}

__global__ void get_decay_coefficient_kernel(float*decay_matrix,float*compensate_matrix,float*decay_coefficient,int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;
    float min = DBL_MAX;
    float temp = 0.;
    for(int i=0;i<ind_size;++i){
        temp = decay_matrix[i*ind_size+idx]/compensate_matrix[i*ind_size+idx];
        if(temp<min){
            min = temp;
        }
    }
    decay_coefficient[idx] = min;

}


__global__ void get_cate_scores_update_kernel(float*cate_scores,float*decay_coefficient,int ind_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>ind_size) return;

    cate_scores[idx] *= decay_coefficient[idx];
}

void matrix_nms_gpu(float*seg_masks,float*cate_labels, float*cate_scores,float*sum_masks,int ind_size){

    const int h_w = mask_pred_shape[1]*mask_pred_shape[2];
    const int wA = h_w;
    const int hA = ind_size;

    float*g_inter_matrix;
    cudaMalloc((void**)&g_inter_matrix,ind_size*ind_size*sizeof(float));

    float*g_iou_matrix;
    cudaMalloc((void**)&g_iou_matrix,ind_size*ind_size*sizeof(float));

    float*g_label_matrix;
    cudaMalloc((void**)&g_label_matrix,ind_size*ind_size*sizeof(float));

    float*g_decay_iou;
    cudaMalloc((void**)&g_decay_iou,ind_size*ind_size*sizeof(float));

    float*g_temp_max;
    cudaMalloc((void**)&g_temp_max,ind_size*sizeof(float));

    float*g_compensate_iou;
    cudaMalloc((void**)&g_compensate_iou,ind_size*ind_size*sizeof(float));

    float*g_decay_matrix;
    cudaMalloc((void**)&g_decay_matrix,ind_size*ind_size*sizeof(float));

    float*g_compensate_matrix;
    cudaMalloc((void**)&g_compensate_matrix,ind_size*ind_size*sizeof(float));

    float*g_decay_coefficient;
    cudaMalloc((void**)&g_decay_coefficient,ind_size*sizeof(float));

    dim3 mygird((hA+BLOCK_DIM-1)/BLOCK_DIM,(hA+BLOCK_DIM-1)/BLOCK_DIM);    
    dim3 myBlock(BLOCK_DIM,BLOCK_DIM);
    get_inter_matrix_kernel<<<mygird,myBlock>>>(seg_masks,g_inter_matrix,hA,wA,hA);

    cudaDeviceSynchronize();

    const int iou_blockSize = 512;
    const int iou_gridSize = (ind_size + iou_blockSize - 1) / iou_blockSize;
    get_iou_matrix_kernel<<<iou_gridSize,iou_blockSize>>>(g_inter_matrix,sum_masks,g_iou_matrix,ind_size);

    get_label_matrix_kernel<<<iou_gridSize,iou_blockSize>>>(cate_labels,g_label_matrix,ind_size);

    cudaDeviceSynchronize();

    get_decay_iou_kernel<<<iou_gridSize,iou_blockSize>>>(g_iou_matrix,g_label_matrix,g_decay_iou,ind_size);

    cudaDeviceSynchronize();

    get_max_kernel<<<iou_gridSize,iou_blockSize>>>(g_decay_iou,g_temp_max,ind_size);

    cudaDeviceSynchronize();

    get_compensate_iou_kernel<<<iou_gridSize,iou_blockSize>>>(g_temp_max,g_compensate_iou,ind_size);

    cudaDeviceSynchronize();

    const int decay_compensate_blockSize = 512;
    const int decay_compensate_gridSize = (ind_size*ind_size + decay_compensate_blockSize - 1) / decay_compensate_blockSize;

    get_decay_compensate_maxtrix_kernel<<<decay_compensate_gridSize,decay_compensate_blockSize>>>(g_decay_iou,g_compensate_iou,
                                        g_decay_matrix,g_compensate_matrix,ind_size*ind_size);

    cudaDeviceSynchronize();
    get_decay_coefficient_kernel<<<iou_gridSize,iou_blockSize>>>(g_decay_matrix,g_compensate_matrix,g_decay_coefficient,ind_size);


    cudaDeviceSynchronize();

    get_cate_scores_update_kernel<<<iou_gridSize,iou_blockSize>>>(cate_scores,g_decay_coefficient,ind_size);

    cudaDeviceSynchronize();

    cudaFree(g_inter_matrix);
    cudaFree(g_iou_matrix);
    cudaFree(g_label_matrix);
    cudaFree(g_decay_iou);
    cudaFree(g_temp_max);
    cudaFree(g_compensate_iou);
    cudaFree(g_decay_matrix);
    cudaFree(g_compensate_matrix);
    cudaFree(g_decay_coefficient);

}

int postprocessing_gpu(float* mask_pred, float*cate_pred,float* kernel_pred,
    float*&seg_preds_cpu,float*&cate_labels_cpu,float*&cate_scores_cpu){
    int ind_size =0 ;
    //int*g_ind_size;
    float *g_cate_scores;
    float*g_cate_label;
    float*g_kernel;

    cudaError err;
    //cudaMalloc((void**)&g_ind_size,1*sizeof(int));
    cudaMalloc((void**)&g_cate_scores,cate_pred_size*sizeof(float));
    cudaMalloc((void**)&g_cate_label,cate_pred_size*sizeof(float));
    cudaMalloc((void**)&g_kernel,kernel_pred_size*sizeof(float));

    

    ind_size= get_cate_ind_gpu(cate_pred,kernel_pred,g_cate_scores,g_cate_label,g_kernel);

    if(ind_size==0){

        cudaFree(g_cate_scores);
        cudaFree(g_cate_label);
        cudaFree(g_kernel);

        return ind_size;
    }


    float*g_seg_dst;
    float*g_seg_masks;
    float*g_sum_masks;
    
    const int h_w = mask_pred_shape[1]*mask_pred_shape[2];
    cudaMalloc((void**)&g_seg_dst,ind_size*h_w*sizeof(float));
    cudaMalloc((void**)&g_seg_masks,ind_size*h_w*sizeof(float));
    cudaMalloc((void**)&g_sum_masks,ind_size*sizeof(float));
    get_seg_preds_conv_gpu(mask_pred,g_kernel,ind_size,g_seg_dst,g_sum_masks,g_seg_masks);
    cudaFree(g_kernel);
    get_cate_scores_gpu(g_seg_dst,g_seg_masks,g_sum_masks,g_cate_scores,ind_size);

    float*sort_cate_labels;
    float*sort_sum_masks;
    float*sort_seg_preds;
    float*sort_seg_masks;
    
    argsort_all_need_gpu(g_cate_scores,g_seg_masks,g_seg_dst,g_sum_masks,g_cate_label,ind_size,sort_cate_labels,sort_sum_masks,sort_seg_preds,sort_seg_masks);
    cudaFree(g_seg_dst);
    cudaFree(g_cate_label);

    start_time = std::chrono::high_resolution_clock::now();
    matrix_nms_gpu(sort_seg_masks,sort_cate_labels,g_cate_scores,sort_sum_masks,ind_size);


    cate_scores_cpu = (float*)calloc(ind_size,sizeof(float));
    cate_labels_cpu = (float*)calloc(ind_size,sizeof(float));
    seg_preds_cpu = (float*)calloc(ind_size*h_w,sizeof(float));

    start_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(cate_labels_cpu,sort_cate_labels,ind_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(cate_scores_cpu,g_cate_scores,ind_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(seg_preds_cpu,sort_seg_preds,ind_size*h_w*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(sort_seg_preds);
    cudaFree(sort_seg_masks);
    cudaFree(g_seg_masks);   
    cudaFree(g_cate_scores);
    cudaFree(g_sum_masks);
    cudaFree(sort_cate_labels);
    cudaFree(sort_sum_masks);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if(err!=cudaSuccess){
        std::cout<<"cuda err "<<cudaGetErrorString(err)<<std::endl;
    }

    return ind_size;

}