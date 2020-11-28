#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include<string>
#include<iostream>
#include<NvInfer.h>
#include<cstdlib>
#include<memory>
#include<fstream>
#include<vector>
#include<sstream>
#include<opencv2/opencv.hpp>


namespace Tn{
    using namespace std;


    enum class RUN_MODE{
        FLOAT32 = 0,
        FLOAT16=1,
    };
    struct InferDeleter{
        template <typename T>
        void operator()(T* obj) const{
            if (obj){
                obj->destroy();
            }
        }
    };


    class onnx2tensorrt{
        template <typename T>
        using nvUniquePtr = unique_ptr<T,InferDeleter>;
    public:
        onnx2tensorrt(string &onnxfFile,int maxBatchSize,RUN_MODE mode=RUN_MODE::FLOAT16);
        onnx2tensorrt(string &enginfFile);
        ~onnx2tensorrt();
        onnx2tensorrt()=delete;
        void infer(const cv::Mat &img,vector<void*> outputData);

        int infer_gpupost(const  cv::Mat &img,float*&seg_preds_cpu,float*&cate_labels_cpu,float*&cate_scores_cpu);
        void saveEngine(string& filename);
        vector<void *> mCudaBuffers;
        vector<size_t> mBindBufferSizes;
        cudaStream_t mCudaStream;
        void* mCudaImg;

    private:
        nvinfer1::Dims mInputDims;
        shared_ptr<nvinfer1::IExecutionContext> mContext;
        shared_ptr<nvinfer1::ICudaEngine> mEngine;
        void initEngine();
};

}


#endif