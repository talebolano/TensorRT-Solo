#include "trt.h"
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include<NvOnnxConfig.h>
#include<fstream>
#include<cuda.h>
#include<NvInfer.h>
#include<NvOnnxParser.h>
#include<numeric>
#include<algorithm>
#include "resize.h"
#include "util.h"
#include<NvInferPlugin.h>
#include"postprocess_gpu.h"

static Logger gLogger;

#define CHECK(status)                                                               \                                           
    do                                                                              \                                          
    {                                                                               \                                        
        auto ret = (status);                                                        \                                        
        if (ret != 0)                                                               \                                       
        {                                                                           \                                      
            std::cerr << "Cuda failure: " << ret << std::endl;                      \                                     
            abort();                                                                \                                    
        }                                                                           \                                   
                                                                                    \
    } while (0)                                                                     

#define CUDA_CHECK(callstr)                                                          \
{                                                                                    \
    cudaError_t error_code = callstr;                                                \
    if(error_code!=cudaSuccess){                                                     \
        std::cerr<<"CUDA err"<<cudaGetErrorString(error_code)<<"at"<<__FILE__<<":"<<__LINE__; \
        assert(0);                                                                   \
    }                                                                                \
}                                                                                    

namespace Tn{

    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        }
        throw std::runtime_error("Invalid DataType.");
        return 0;
    }
    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    
    inline void* safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr){
            std::cerr << "Out of memory" << std::endl;
            exit(1);
        }
        return deviceMem;
    }

    onnx2tensorrt::onnx2tensorrt(std::string &onnxfFile,int maxBatchSize,Tn::RUN_MODE mode){
        cudaSetDevice(0);
        auto builder = nvUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        if(!builder){
            exit(EXIT_FAILURE);
        }

        const auto explicitBatch = 1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
        auto network = nvUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if(!network){
            exit(EXIT_FAILURE);
        }
        auto config = nvUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if(!config){
            exit(EXIT_FAILURE);
        }
        auto parser = nvUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network,gLogger.getTRTLogger()));
        if(!parser){
            exit(EXIT_FAILURE);
        }
        auto parsed = parser->parseFromFile(
            onnxfFile.c_str(),
            static_cast<int>(gLogger.getReportableSeverity())
        );
        if(!parsed){
            std::cout<<"failed to parse from "<<onnxfFile<<"!"<<std::endl;
            exit(EXIT_FAILURE);
        }

        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(1<<30);
        if(mode==RUN_MODE::FLOAT16){
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout<<"FP16 mode "<<std::endl;
        }else std::cout<<"FP32 mode "<<std::endl;

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network,*config),InferDeleter());


        if(!mEngine)
        {
            exit(EXIT_FAILURE);
        }
        assert(network->getNbInputs()==1);
        mInputDims = network->getInput(0)->getDimensions();

        assert(mInputDims.nbDims==4);

        assert(network->getNbOutputs()==3);
        std::cout<<"haved genrated engine"<<std::endl;
        initEngine();
        }

    onnx2tensorrt::onnx2tensorrt(std::string &enginfFile){

        cudaSetDevice(0);
        std::fstream file;
        file.open(enginfFile,std::ios::binary|std::ios::in);
        if(!file.is_open()){
            std::cout<<"read engine "<<enginfFile<<" failed"<<std::endl;
            return;
        }
        file.seekg(0,ios::end);
        int length = file.tellg();
        file.seekg(0,std::ios::beg);
        unique_ptr<char[]> data(new char[length]);
        file.read(data.get(),length);
        file.close();
        initLibNvInferPlugins(&gLogger,"");
        auto mRuntime = nvUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
        if(!mRuntime)
        {
            exit(EXIT_FAILURE);
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(data.get(),length),InferDeleter());
        if(!mEngine)
        {
            exit(EXIT_FAILURE);
        }
        
        initEngine();
        std::cout<<"load genrated engine"<<std::endl;
    }


    void onnx2tensorrt::saveEngine(std::string& filename){
        if(mEngine)
        {
            shared_ptr<nvinfer1::IHostMemory> data(mEngine->serialize(),InferDeleter());
            std::ofstream file;
            file.open(filename,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout<<"read create file failed"<<std::endl;
                file.close();
            }
            file.write((const char *)data->data(),data->size());
            file.close();
        }
        return;
    }

    void onnx2tensorrt::initEngine(){
        const int maxBatchSize = mEngine->getMaxBatchSize();
        mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(),InferDeleter());

        assert(mContext!=nullptr);
        int nbBindings = mEngine->getNbBindings();
        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(nbBindings); 

        for(int i=0;i<nbBindings;++i){
            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            int totalSize = maxBatchSize*volume(mEngine->getBindingDimensions(i))*getElementSize(dtype);//

            mBindBufferSizes[i] = totalSize;
            mCudaBuffers[i] = safeCudaMalloc(totalSize);     
        }
        mCudaImg = safeCudaMalloc(4096*4096*3*sizeof(uchar));
        CUDA_CHECK(cudaStreamCreate(&mCudaStream));
        mInputDims = mEngine->getBindingDimensions(0);

        return;
    }


    void onnx2tensorrt::infer(const  cv::Mat &img,vector<void*> outputData){
        bool keepRation=0,keepCenter=0;
        CUDA_CHECK(cudaMemcpy(mCudaImg,img.data,img.step[0]*img.rows,cudaMemcpyHostToDevice));

        resizeAndNorm(mCudaImg,(float*)mCudaBuffers[0],img.cols,img.rows,mInputDims.d[3],mInputDims.d[2],keepRation,keepCenter,mCudaStream);

        mContext->executeV2(&mCudaBuffers[0]);

        for(size_t bindindIdx=1;bindindIdx<mBindBufferSizes.size();bindindIdx++){   
            auto size = mBindBufferSizes[bindindIdx];
            CUDA_CHECK(cudaMemcpy(outputData[bindindIdx-1],mCudaBuffers[bindindIdx],size,cudaMemcpyDeviceToHost));//

        }

    }


    int onnx2tensorrt::infer_gpupost(const cv::Mat &img,float*&seg_preds_cpu,float*&cate_labels_cpu,float*&cate_scores_cpu){
        bool keepRation=0,keepCenter=0;
        int ind_size;
        CUDA_CHECK(cudaMemcpy(mCudaImg,img.data,img.step[0]*img.rows,cudaMemcpyHostToDevice));
        resizeAndNorm(mCudaImg,(float*)mCudaBuffers[0],img.cols,img.rows,mInputDims.d[3],mInputDims.d[2],keepRation,keepCenter,mCudaStream);
        mContext->executeV2(&mCudaBuffers[0]);
        ind_size = postprocessing_gpu((float*)mCudaBuffers[1],(float*)mCudaBuffers[2],(float*)mCudaBuffers[3],
                            seg_preds_cpu,cate_labels_cpu,cate_scores_cpu);

        return ind_size;

    }

    onnx2tensorrt::~onnx2tensorrt(){
        cudaStreamSynchronize(mCudaStream);
        cudaStreamDestroy(mCudaStream);
        mContext.reset();
        for(size_t bindindIdx=0;bindindIdx<mBindBufferSizes.size();++bindindIdx){
            if(mCudaBuffers[bindindIdx])CUDA_CHECK(cudaFree(mCudaBuffers[bindindIdx]));

        }
        if(mCudaImg)CUDA_CHECK(cudaFree(mCudaImg));
    }




}