#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "config.h"
#include "cublas_v2.h"
#include "seg_conv.h"
#include <chrono>
#include <iostream>

#define BLOCK_DIM 32


__device__ float sigmiod(float src){
    
    return 1./(1.+expf(-src));

}

__global__ void get_conv2_kernel(float*kernel,float*src,float* dst,float*seg_masks,int hA,int wA,int wB){
    

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
        dst[c+wB*ty+tx] = sigmiod(dst[c+wB*ty+tx]);
        if(dst[c+wB*ty+tx]>0.5){
            seg_masks[c+wB*ty+tx] = 1.;
        }
        else{
            seg_masks[c+wB*ty+tx] = 0.;
        }
        
    }



}


int get_seg_preds_conv2_gpu(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks){
    const int h_w = mask_pred_shape[1]*mask_pred_shape[2];
    const int wB = h_w;
    const int hA = ind_size;

    float*g_seg_preds;
    float*g_kernels;
    float*g_dst;
    float*g_seg_masks;

    cudaMalloc((void**)&g_seg_preds,mask_pred_size*sizeof(float));
    cudaMalloc((void**)&g_kernels,ind_size*kernel_pred_shape[1]*sizeof(float));
    cudaMalloc((void**)&g_dst,ind_size*h_w*sizeof(float));
    cudaMalloc((void**)&g_seg_masks,ind_size*h_w*sizeof(float));

    cudaMemcpy(g_seg_preds,seg_preds,sizeof(float)*mask_pred_size,cudaMemcpyHostToDevice);
    cudaMemcpy(g_kernels,kernels,sizeof(float)*ind_size*kernel_pred_shape[1],cudaMemcpyHostToDevice);
    dim3 mygird((wB+BLOCK_DIM-1)/BLOCK_DIM,(hA+BLOCK_DIM-1)/BLOCK_DIM);
    
    dim3 myBlock(BLOCK_DIM,BLOCK_DIM);

    get_conv2_kernel<<<mygird,myBlock>>>(g_kernels,g_seg_preds,g_dst,g_seg_masks,hA,kernel_pred_shape[1],wB);
    cudaMemcpy(dst,g_dst,ind_size*h_w*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(seg_masks,g_seg_masks,ind_size*h_w*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(g_seg_preds);
    cudaFree(g_kernels);
    cudaFree(g_dst);
    cudaFree(g_seg_masks);

    for(int i=0;i<ind_size;++i){
        for(int hw=0;hw<h_w;++hw){
            sum_masks[i]+=seg_masks[i*h_w+hw];
        }
    }


    return 0;
}


int get_seg_preds_conv2_cublas(float*seg_preds,float*kernels,int ind_size,float*dst,
                        float*sum_masks,float*seg_masks){

    cublasStatus_t status;
    const int h_w = mask_pred_shape[1]*mask_pred_shape[2];
    const int wB = h_w;
    const int hA = ind_size;

    float*g_seg_preds;
    float*g_kernels;
    float*g_dst;
    //float*g_seg_masks;

    cudaMalloc((void**)&g_seg_preds,mask_pred_size*sizeof(float));
    cudaMalloc((void**)&g_kernels,ind_size*kernel_pred_shape[1]*sizeof(float));
    cudaMalloc((void**)&g_dst,ind_size*h_w*sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMemcpy(g_seg_preds,seg_preds,sizeof(float)*mask_pred_size,cudaMemcpyHostToDevice);
    cudaMemcpy(g_kernels,kernels,sizeof(float)*ind_size*kernel_pred_shape[1],cudaMemcpyHostToDevice);

    const float a=1,b=0;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        wB,
        hA,
        kernel_pred_shape[1],
        &a,
        g_seg_preds, // b^t
        wB,
        g_kernels, //a^t
        kernel_pred_shape[1],
        &b,
        g_dst,
        wB
    );

    cudaMemcpy(dst,g_dst,ind_size*h_w*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(g_seg_preds);
    cudaFree(g_kernels);
    cudaFree(g_dst);

    for(int i=0;i<ind_size;++i){
        for(int hw=0;hw<h_w;++hw){
            dst[i*h_w+hw] = 1./(1.+exp(-dst[i*h_w+hw])); //sigmoid
            if (dst[i*h_w+hw]>0.5){
                seg_masks[i*h_w+hw] = 1.;
                sum_masks[i]+=1;
            }
        }
    }

    return 0;
}
